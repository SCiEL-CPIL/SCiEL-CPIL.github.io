"""Greedy + local-search covering code construction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def _base_to_index(word, n, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _construct_greedy(q, n, radius, deadline, init_seed=None):
    total = q ** n
    if total > 60_000_000:
        return None  # too large for dense coverage array

    # Precompute all words as int arrays
    words = np.array(
        list(itertools.product(range(q), repeat=n)),
        dtype=np.uint8,
    )
    if words.size == 0:
        words = words.reshape(0, n)

    # Precompute neighbor index lists for each word (Hamming distance <= radius)
    # For each position and each alternative symbol, generate the neighbor index.
    # We build a list of arrays: neigh[i] = np.array of indices within radius of word i.
    # Efficient approach: for radius 1, neighbors = self + q-1 (one per position per alt).
    # We'll compute per-word neighbor index arrays lazily but cache the "delta" structure.

    # Build a mapping: for each position p and symbol change, the additive delta in index.
    # index = sum_k word[k] * q^(n-1-k). Changing position p by delta changes index by delta * q^(n-1-p).
    pow_q = [1]
    for _ in range(n):
        pow_q.append(pow_q[-1] * q)
    # pow_q[k] = q^k ; position p (0-based) has weight q^(n-1-p)
    pos_weight = [pow_q[n - 1 - p] for p in range(n)]

    # For radius 1: neighbors of word i are i and i +/- pos_weight[p] for each p and each alt symbol.
    # We'll generate candidate neighbor index arrays on the fly in a vectorized way per candidate.

    # Precompute the set of "offsets" for radius 1: for each position p, the list of index
    # deltas corresponding to changing that position to any other symbol. But the delta depends
    # on the current symbol value at that position. So we need per-word offsets.

    # Alternative efficient scheme: precompute for each word i a flat array of neighbor indices.
    # For n=6, q=3, radius=1: each word has 1 + 6*2 = 13 neighbors.
    # Build neighbors as a list of int arrays.
    neigh = []
    for i in range(total):
        w = words[i]
        idx = i
        arr = [idx]
        for p in range(n):
            cur = w[p]
            wt = pos_weight[p]
            for s in range(q):
                if s == cur:
                    continue
                arr.append(idx - cur * wt + s * wt)
        neigh.append(np.array(arr, dtype=np.int64))

    # Coverage array
    covered = np.zeros(total, dtype=np.bool_)
    codewords = []

    # Initialize: optionally seed with a few codewords
    if init_seed is not None:
        for cw in init_seed:
            idx = _base_to_index(cw, n, q)
            if not covered[idx]:
                covered[neigh[idx]] = True
                codewords.append(idx)

    # Greedy
    # We maintain a count of uncovered neighbors for each candidate.
    # uncovered_count[i] = number of j in neigh[i] with not covered[j].
    # Recompute periodically or incrementally.
    # For efficiency, recompute from scratch each iteration using vectorized ops is O(total*|neigh|)
    # which for 729*13 ~ 9477 per iteration is fine.

    while True:
        if time.time() > deadline:
            break
        # Compute uncovered count for each candidate
        # covered is bool array; for each i, count of neigh[i] not in covered.
        # Vectorized: build a 2D view is expensive; instead loop but it's only 729.
        # Use: for each i, uncovered_count[i] = len(neigh[i]) - covered[neigh[i]].sum()
        # We'll compute via a loop over i (total=729, cheap).
        best_i = -1
        best_cnt = -1
        for i in range(total):
            if covered[i]:
                # already covered as a codeword position; still could be a candidate but skip
                # Actually we can still pick it? No, picking a covered codeword position is redundant.
                continue
            cnt = len(neigh[i]) - int(covered[neigh[i]].sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
        if best_i < 0 or best_cnt <= 0:
            break
        covered[neigh[best_i]] = True
        codewords.append(best_i)

    return codewords


def _local_search(codewords, q, n, radius, deadline):
    total = q ** n
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    pow_q = [1]
    for _ in range(n):
        pow_q.append(pow_q[-1] * q)
    pos_weight = [pow_q[n - 1 - p] for p in range(n)]

    # Build neighbor arrays
    neigh = []
    for i in range(total):
        w = words[i]
        idx = i
        arr = [idx]
        for p in range(n):
            cur = w[p]
            wt = pos_weight[p]
            for s in range(q):
                if s == cur:
                    continue
                arr.append(idx - cur * wt + s * wt)
        neigh.append(np.array(arr, dtype=np.int64))

    cw = list(codewords)
    # covered count per cell: number of codewords whose neighborhood contains this cell
    count = np.zeros(total, dtype=np.int32)
    for i in cw:
        count[neigh[i]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # 1) Remove redundant codewords
        for k in range(len(cw) - 1, -1, -1):
            if time.time() > deadline:
                break
            i = cw[k]
            if (count[neigh[i]] == 1).all():
                count[neigh[i]] -= 1
                cw.pop(k)
                improved = True

        # 2) Swap: replace one codeword with another to allow a removal
        # For each codeword i, try replacing with j not in cw, check if count stays >=1 everywhere.
        # This is expensive; do a limited pass.
        if len(cw) > 0:
            cw_set = set(cw)
            # For each candidate removal i, find a j that covers all cells that only i covers.
            for k in range(len(cw)):
                if time.time() > deadline:
                    break
                i = cw[k]
                # cells only covered by i
                only_i = np.where(count == 1)
                # We need a j such that neigh[j] covers all of only_i.
                # Find the set of cells that would become uncovered if we remove i.
                # If empty, we already removed it in step 1.
                # Try swapping: remove i, add j.
                # For efficiency, just try a few j candidates.
                # Find j that maximizes overlap with only_i.
                # This is O(total) per i, O(len(cw)*total) per pass.
                only_i_set = set(only_i[0].tolist())
                if not only_i_set:
                    continue
                # Try each j not in cw
                best_j = -1
                best_gain = 0
                # Limit search
                for j in range(total):
                    if j in cw_set:
                        continue
                    # gain = number of only_i cells covered by j
                    nj = neigh[j]
                    gain = 0
                    for c in only_i_set:
                        if covered_by := False:
                            pass
                    # Vectorized: check if all only_i are in neigh[j]
                    # Build a set of neigh[j]
                    nj_set = set(nj.tolist())
                    # Count how many only_i are in nj_set
                    gain = sum(1 for c in only_i_set if c in nj_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                    if gain == len(only_i_set):
                        break
                if best_j >= 0 and best_gain == len(only_i_set):
                    # Perform swap
                    count[neigh[i]] -= 1
                    count[neigh[best_j]] += 1
                    cw[k] = best_j
                    cw_set.discard(i)
                    cw_set.add(best_j)
                    improved = True

        # 3) Add a codeword if it enables a removal of two others (net -1)
        # Skip for now to keep it simple; the above should suffice.

    return cw


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin

    total = q ** n
    if total > 60_000_000:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Greedy construction
    codewords = _construct_greedy(q, n, radius, deadline)
    if codewords is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Local search
    codewords = _local_search(codewords, q, n, radius, deadline)

    # Convert to list of lists
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    result = [words[i].tolist() for i in codewords]

    return result
# EVOLVE-BLOCK-END
