"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0  # wall-clock safety margin under the 150s kill
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _elapsed()

def construct(q, n, radius):
    _START[0] = time.time()
    # We solve the concrete task q=3, n=8, radius=1 (the evaluator passes these).
    # For generality we fall back to a trivial full-space cover if parameters differ.
    if (q, n, radius) != (3, n, radius) or q != 3 or n != 8 or radius != 1:
        # Trivial valid cover: every word is a codeword (distance 0).
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)

    # ---- Build the universe of all 6561 words and their radius-1 balls ----
    N = 3 ** n  # 6561
    # Encode each word as an integer 0..6560 via base-3.
    # Precompute, for each word index, the set of words within distance <=1.
    # We'll work with integer encodings for speed.

    # word index <-> base-3 digits
    # digits of index i in base 3, length n (little-endian position 0 = least significant)
    def idx_to_digits(i):
        d = [0] * n
        for p in range(n):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(n):
            v += d[p] * mul
            mul *= 3
        return v

    # For each word, precompute its closed ball (itself + single-position changes).
    # ball[i] = list of indices within distance <=1 of word i.
    ball = np.empty(N, dtype=object)
    for i in range(N):
        d = idx_to_digits(i)
        s = [i]
        for p in range(n):
            for nv in range(3):
                if nv != d[p]:
                    d[p] = nv
                    s.append(digits_to_idx(d))
                    d[p] = idx_to_digits(i)[p]
        ball[i] = np.array(s, dtype=np.int32)

    # ---- Initial construction: greedy seed ----
    # Start with a reasonable initial set, then greedy-fill, then local search.

    # Initial: use a product of tetracode-style structure to get a decent start.
    # Tetracode: 9 codewords length 4, perfect radius-1.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Build a seed set: for each tetracode word on coords 0-3, place a tetracode word
    # on coords 4-7. This gives 9*9=81 words, but only covers radius-1 *within* each half.
    # We will then greedily add codewords to cover the rest, and locally optimize.

    def digits_to_idx2(d):
        v = 0
        mul = 1
        for p in range(n):
            v += d[p] * mul
            mul *= 3
        return v

    seed_set = []
    for h in tetracode:
        for t in tetracode:
            w = h + t
            seed_set.append(digits_to_idx2(w))
    seed_set = list(set(seed_set))

    # ---- Greedy set cover to reach full coverage ----
    covered = np.zeros(N, dtype=bool)
    chosen = []

    # Mark coverage from seed
    for i in seed_set:
        covered[ball[i]] = True
    chosen = list(seed_set)

    # Greedy: repeatedly add the codeword covering the most *new* (uncovered) words.
    # To be fast, maintain for each candidate the count of uncovered words it covers.
    # We'll do a few greedy passes.

    def greedy_fill(chosen, covered, max_rounds=4000):
        # uncovered_count per candidate
        # We recompute incrementally: for each candidate i, new_cover = |ball[i] ∩ uncovered|
        # Maintain uncovered as a set for membership, but for counting use covered array.
        for _ in range(max_rounds):
            if _time_left() < 20.0:
                break
            # find best candidate
            best_i = -1
            best_gain = 0
            # Iterate over all N candidates; count uncovered in its ball.
            # Use numpy for speed.
            for i in range(N):
                if covered[i]:
                    # candidate already covered; still could add new words, but skip if fully covered ball
                    pass
                b = ball[i]
                gain = int(np.sum(~covered[b]))
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_gain <= 0:
                break
            chosen.append(best_i)
            covered[ball[best_i]] = True
        return chosen, covered

    chosen, covered = greedy_fill(chosen, covered)

    if not covered.all():
        # safety: add remaining uncovered as codewords (distance 0 guarantees cover)
        for i in range(N):
            if not covered[i]:
                chosen.append(i)
                covered[ball[i]] = True
        if _time_left() < 5.0:
            return [idx_to_digits(i) for i in chosen]

    # ---- Local search: try to remove a codeword if its ball is fully covered by others ----
    # Represent chosen as a set for O(1) removal checks.
    def local_search_remove(chosen, covered, max_iter=200000):
        chosen_set = set(chosen)
        # coverage count per word: how many chosen codewords cover it
        covcount = np.zeros(N, dtype=np.int32)
        for i in chosen:
            covcount[ball[i]] += 1
        # A codeword i can be removed if every word in ball[i] has covcount >= 2.
        # i.e., min covcount over ball[i] >= 2.
        removed_any = True
        it = 0
        while removed_any and _time_left() > 15.0 and it < max_iter:
            it += 1
            removed_any = False
            # iterate candidates in a random-ish order
            order = list(chosen_set)
            rng.shuffle(order)
            for i in order:
                if i not in chosen_set:
                    continue
                b = ball[i]
                if int(np.min(covcount[b])) >= 2:
                    # safe to remove
                    covcount[b] -= 1
                    chosen_set.discard(i)
                    removed_any = True
        return list(chosen_set), covcount

    chosen, covcount = local_search_remove(chosen, covered)

    # ---- Local search: swap — remove one codeword, then greedily add fewer to re-cover ----
    # Try: for a candidate removal, see if the resulting uncovered set can be covered
    # by adding <=1 codeword (net reduction). We'll do limited swaps.
    def local_search_swap(chosen, covcount, max_iter=50000):
        chosen_set = set(chosen)
        improved = True
        it = 0
        while improved and _time_left() > 12.0 and it < max_iter:
            it += 1
            improved = False
            order = list(chosen_set)
            rng.shuffle(order)
            for i in order:
                if i not in chosen_set:
                    continue
                b = ball[i]
                # words that would become uncovered if we remove i
                would_uncover = b[covcount[b] == 1]
                if len(would_uncover) == 0:
                    # pure removal
                    covcount[b] -= 1
                    chosen_set.discard(i)
                    improved = True
                    continue
                # need a replacement covering all of would_uncover
                # find a codeword j (not in set) whose ball covers all would_uncover
                # and that is "cheaper" (net 0 or -1). Since we remove 1 and add 1,
                # net 0; only do if it lets a later removal happen. To keep simple,
                # only accept if adding j covers would_uncover AND j is not i.
                # We'll accept a swap that keeps size same only if it enables progress;
                # to limit cost, accept swaps that reduce a "potential".
                # Simplify: accept a swap only if the new codeword's ball covers
                # would_uncover and also covers at least one word currently covered
                # only by i in a way that... (complex). Instead, do a 1-for-1 swap
                # only if it strictly reduces the number of words with covcount==1
                # (tighter packing) — a heuristic potential.
                target = set(would_uncover.tolist())
                # search for j
                found = -1
                for j in range(N):
                    if j in chosen_set:
                        continue
                    bj = ball[j]
                    if target.issubset(set(bj.tolist())):
                        # check potential improvement
                        # new covcount after remove i, add j
                        # words losing coverage: b with covcount==1 -> -1 (these are would_uncover, covered by j so +1 back)
                        # words gaining: bj with covcount==0 -> +1
                        gain_zero = int(np.sum(covcount[bj] == 0))
                        # would_uncover lose 1 then gain 1 (net 0) for those in bj
                        # net change in count of covcount==1 words:
                        # remove i: words in b with covcount==1 become 0 -> count of ==1 decreases by len(would_uncover)
                        # add j: words in bj with covcount==0 become 1 -> count of ==1 increases by gain_zero
                        # words in bj with covcount==1 become 2 -> count of ==1 decreases by (# in bj with covcount==1)
                        ones_in_bj = int(np.sum(covcount[bj] == 1))
                        delta_potential = -len(would_uncover) + gain_zero - ones_in_bj
                        if delta_potential < 0:
                            found = j
                            break
                if found >= 0:
                    bj = ball[found]
                    covcount[b] -= 1
                    covcount[bj] += 1
                    chosen_set.discard(i)
                    chosen_set.add(found)
                    improved = True
        return list(chosen_set), covcount

    chosen, covcount = local_search_swap(chosen, covcount)

    # ---- Final verification (cheap) and return ----
    # Ensure full coverage; if not, patch.
    final_covered = np.zeros(N, dtype=bool)
    for i in chosen:
        final_covered[ball[i]] = True
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen.append(i)
                final_covered[ball[i]] = True

    # Deduplicate
    chosen = list(dict.fromkeys(chosen))

    return [idx_to_digits(i) for i in chosen]
