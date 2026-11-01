"""Greedy covering-code search for K_q(n, radius) with a wall-clock guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 105.0  # safety margin well under the 150s kill

    # Special case: a perfect code exists (q**(n-1) words cover exactly).
    # For q=3,n=4,r=1 the ternary tetracode (9 words) is perfect.
    if q ** (n - 1) == q ** n // q and _is_perfect_case(q, n, radius):
        return _perfect_code(q, n, radius)

    # Build the "neighbourhood" table: for each word, the set of words within
    # Hamming distance `radius` (as a set of base-q indices).
    # For radius=1: the word itself plus one change per coordinate per other symbol.
    # For general small radius we enumerate via itertools.
    import itertools
    qn = q ** n
    neigh = []
    # Precompute single-symbol deltas for radius>=1 to speed up radius=1.
    if radius == 1:
        # For each index, compute its neighbour indices.
        # Convert index to digits lazily via vectorised approach for speed.
        # Precompute digits of all words as an array of shape (qn, n).
        digits = np.empty((qn, n), dtype=np.int32)
        for i in range(qn):
            v = i
            for c in range(n - 1, -1, -1):
                digits[i, c] = v % q
                v //= q
        # For each word, neighbours = itself + for each coord, change to each other symbol.
        # Precompute powers.
        powq = [1] * (n + 1)
        for c in range(1, n + 1):
            powq[c] = powq[c - 1] * q
        neigh = [None] * qn
        for i in range(qn):
            s = set()
            s.add(i)
            d = digits[i]
            for c in range(n):
                base = d[c]
                p = powq[c]
                for sym in range(q):
                    if sym == base:
                        continue
                    s.add(i + (sym - base) * p)
            neigh[i] = s
    else:
        # General radius: enumerate all words and compute Hamming distance to i.
        digits = np.empty((qn, n), dtype=np.int32)
        for i in range(qn):
            v = i
            for c in range(n - 1, -1, -1):
                digits[i, c] = v % q
                v //= q
        # Hamming distance matrix (qn x qn) is too big for large qn; but qn<=2187 so ok.
        # Compute distance via broadcasting.
        # dist[i,j] = number of coords where digits[i]!=digits[j]
        # Use chunking to limit memory.
        dist = np.zeros((qn, qn), dtype=np.int32)
        chunk = 512
        for a0 in range(0, qn, chunk):
            a1 = min(a0 + chunk, qn)
            # (chunk,1,n) vs (1,qn,n)
            diff = (digits[a0:a1, None, :] != digits[None, :, :]).sum(axis=2)
            dist[a0:a1, :] = diff
        neigh = [None] * qn
        for i in range(qn):
            neigh[i] = set(np.where(dist[i] <= radius)[0].tolist())

    # Coverage tracking: count how many chosen codewords cover each word.
    cover_count = np.zeros(qn, dtype=np.int32)
    uncovered = qn
    chosen = []

    def time_left():
        return deadline - time.time()

    def add_word(w):
        nonlocal uncovered
        for x in neigh[w]:
            if cover_count[x] == 0:
                cover_count[x] = 1
                uncovered -= 1
            else:
                cover_count[x] += 1
        chosen.append(w)

    # Phase 1: greedy seed by largest marginal coverage.
    # We'll iterate: pick the word that maximises newly-covered words, until all covered.
    # To be fast, maintain for each candidate word its current marginal gain
    # (number of neighbours with cover_count==0).
    # We recompute gains lazily.
    # Use a priority approach: keep a list of (gain, word). When we add a word,
    # only the neighbours' gains change.
    import heapq
    # gain[w] = number of neighbours x of w with cover_count[x]==0
    gain = np.array([len(neigh[w]) for w in range(qn)], dtype=np.int32)
    # heap of (-gain, word)
    heap = [(-int(gain[w]), w) for w in range(qn)]
    heapq.heapify(heap)
    in_heap_valid = [True] * qn
    # We'll use a lazy heap: pop, check if gain still matches, else skip.
    # Track current gain in a dict/array; when we add a word we update gains of its neighbours.

    while uncovered > 0:
        if time_left() < 2.0:
            break
        # pop until we find a valid entry
        while True:
            negg, w = heapq.heappop(heap)
            if -negg == int(gain[w]):
                break
            # stale entry, skip
            # (we don't push duplicates; we push updated entries when gains change)
        if -negg <= 0:
            # no progress possible via greedy (shouldn't happen if uncovered>0)
            break
        add_word(w)
        # Update gains for all neighbours of w (their marginal gain decreased by 1
        # if they were uncovered, i.e., cover_count was 0 before add).
        # Actually gain[x] counts neighbours of x that are currently uncovered.
        # When we add w, for each neighbour x of w, if x was uncovered (cover_count 0 before),
        # then x becomes covered, so for every y that has x as a neighbour, gain[y] decreases.
        # Simpler: recompute gain for all y that are neighbours of any newly-covered word.
        # Newly covered words = neighbours of w that had cover_count 0 before add.
        # We can detect them: after add, cover_count[x]>=1; those that went 0->1 are the ones
        # where before it was 0. We didn't record, so recompute: for x in neigh[w],
        # the set of y such that x in neigh[y] i.e. y in neigh[x].
        # To limit work, update gain for y in union over newly-covered x of neigh[x].
        # But we don't know which x were newly covered without tracking. Approximate:
        # recompute gain for all y in neigh[w] (the added word's neighbours) plus the added word.
        # This is an approximation that keeps heap mostly consistent; stale entries are skipped.
        for x in neigh[w]:
            for y in neigh[x]:
                # recompute gain[y] exactly (cheap: count uncovered neighbours)
                g = 0
                for z in neigh[y]:
                    if cover_count[z] == 0:
                        g += 1
                gain[y] = g
                heapq.heappush(heap, (-g, y))
            # also the added word itself may have changed gain (it's now covered)
        # recompute gain for w too
        g = 0
        for z in neigh[w]:
            if cover_count[z] == 0:
                g += 1
        gain[w] = g
        heapq.heappush(heap, (-g, w))

    # Ensure we have a valid covering (if greedy didn't finish, finish with a simple pass).
    if uncovered > 0:
        # Simple: add words to cover remaining uncovered.
        # Find uncovered words.
        remaining = [i for i in range(qn) if cover_count[i] == 0]
        for w in remaining:
            if time_left() < 2.0:
                break
            # pick the word w itself (covers itself)
            add_word(w)
        # verify all covered
        if uncovered > 0:
            # force add every uncovered word
            for i in range(qn):
                if cover_count[i] == 0:
                    add_word(i)

    # Phase 2: local search to remove redundant codewords and do 3-for-2 swaps.
    # Represent chosen as a set for O(1) membership.
    chosen_set = set(chosen)
    # For removal: a codeword c is removable if every word it uniquely covers
    # (cover_count==1 and c is the only one) is covered by another chosen word.
    # i.e., for all x in neigh[c], there exists c' in chosen_set, c'!=c, with x in neigh[c'].
    # Equivalently: for all x in neigh[c] with cover_count[x]==1, c is the unique coverer
    #   -> then removing c leaves x uncovered. So c is removable iff no x in neigh[c]
    #   has cover_count[x]==1 (i.e., all neighbours of c have cover_count>=2).
    # Wait: cover_count[x] counts how many chosen words cover x. If cover_count[x]==1,
    #   the unique coverer is c (since c covers x). Removing c -> x uncovered. So c
    #   removable iff for all x in neigh[c], cover_count[x] != 1, i.e., >=2.
    # Actually we need: after removing c, every x in neigh[c] still covered by some other.
    #   x is covered by c and possibly others. If cover_count[x]>=2, then >=1 other covers it. Good.
    #   If cover_count[x]==1, only c covers it -> bad. So condition: all x in neigh[c] have cover_count[x]>=2.
    # That's correct.

    def try_remove(c):
        nonlocal uncovered
        ok = True
        for x in neigh[c]:
            if cover_count[x] <= 1:
                ok = False
                break
        if ok:
            for x in neigh[c]:
                cover_count[x] -= 1
                if cover_count[x] == 0:
                    uncovered += 1
            chosen_set.discard(c)
            chosen.remove(c)
            return True
        return False

    # Greedy removal pass (repeat until no more removable).
    if time_left() > 5.0:
        changed = True
        while changed and time_left() > 5.0:
            changed = False
            # iterate over a copy
            for c in list(chosen):
                if try_remove(c):
                    changed = True

    # Phase 3: 3-for-2 swap / add-remove to escape local minima.
    # Try: pick two codewords to remove and one to add, if it keeps coverage and reduces size.
    # This is expensive; do a limited randomised version within time budget.
    if time_left() > 10.0:
        # Precompute for each word the set of chosen codewords that cover it? Too big.
        # Instead, do random trials:
        #   pick candidate new word w not in chosen_set.
        #   Determine which current codewords become removable after adding w.
        #   If >=2 become removable and w not already present, do the swap.
        rng = random.Random(12345)
        attempts = 0
        while time_left() > 8.0 and attempts < 200000:
            attempts += 1
            if attempts % 2000 == 0 and time_left() < 8.0:
                break
            w = rng.randrange(qn)
            if w in chosen_set:
                continue
            # Temporarily add w.
            # Find codewords c in chosen that are "saturated" only by w among their unique covers.
            # A codeword c becomes removable after adding w iff for all x in neigh[c],
            #   (cover_count[x] >= 2) or (cover_count[x]==1 and w covers x).
            # i.e., the only uncovered-after-removal risk (x with cover_count[x]==1) must be covered by w.
            # Count how many c satisfy this. If >=2, removing them and adding w reduces size by 1.
            removable = []
            for c in chosen:
                # check
                ok = True
                for x in neigh[c]:
                    if cover_count[x] == 1:
                        # x only covered by c; need w to cover x
                        if x not in neigh[w]:
                            ok = False
                            break
                if ok:
                    removable.append(c)
            if len(removable) >= 2:
                # Do the swap: remove two, add w.
                c1, c2 = removable[0], removable[1]
                # remove c1, c2
                for c in (c1, c2):
                    for x in neigh[c]:
                        cover_count[x] -= 1
                    chosen_set.discard(c)
                    chosen.remove(c)
                # add w
                for x in neigh[w]:
                    if cover_count[x] == 0:
                        uncovered -= 1
                    cover_count[x] += 1
                chosen_set.add(w)
                chosen.append(w)
                # After swap, run a quick removal pass.
                for cc in list(chosen):
                    try_remove(cc)
                # Loop continues to try more swaps.

    # Convert chosen indices to codeword digit lists.
    powq = [1] * (n + 1)
    for c in range(1, n + 1):
        powq[c] = powq[c - 1] * q
    result = []
    for w in chosen:
        digits = []
        v = w
        for c in range(n):
            digits.append(v % q)
            v //= q
        result.append(digits)
    return result


def _is_perfect_case(q, n, radius):
    # A perfect covering code of radius r has size q**(n-r) and exists for
    # q=3, n=4, r=1 (ternary tetracode).
    if q == 3 and n == 4 and radius == 1:
        return True
    return False


def _perfect_code(q, n, radius):
    if q == 3 and n == 4 and radius == 1:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Fallback: shouldn't be called.
    return [[0] * n]
# EVOLVE-BLOCK-END
