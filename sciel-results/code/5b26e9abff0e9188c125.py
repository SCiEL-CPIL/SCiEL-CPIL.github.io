"""Covering code K_q(n, radius): randomized construction + local improvement."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

_DEADLINE = 140.0  # safety margin below the 150 s wall-clock kill


def _all_words(q, n):
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words


def _build_neighbours(q, n, radius, words):
    """Precompute neighbour matrix as a 2D numpy array (padded) or list."""
    total = words.shape[0]
    words_int = words.astype(np.int64)
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    if radius == 1:
        nb = np.empty((total, 1 + n * (q - 1)), dtype=np.int64)
        for idx in range(total):
            digits = words_int[idx]
            pos = 0
            nb[idx, pos] = idx
            pos += 1
            for i in range(n):
                v = digits[i]
                for vp in range(q):
                    if vp == v:
                        continue
                    nb[idx, pos] = idx + (vp - v) * stride[i]
                    pos += 1
        return nb
    else:
        # General radius: build as list of sorted lists
        patterns = []
        for k in range(radius + 1):
            for subset in itertools.combinations(range(n), k):
                if k == 0:
                    patterns.append([])
                else:
                    for assign in itertools.product(range(q), repeat=k):
                        patterns.append([(subset[j], assign[j]) for j in range(k)])
        nb_list = []
        for idx in range(total):
            digits = words_int[idx]
            nbs = set()
            for pat in patterns:
                new_idx = idx
                for (coord, new_sym) in pat:
                    new_idx += (new_sym - digits[coord]) * stride[coord]
                if 0 <= new_idx < total:
                    nbs.add(new_idx)
            nb_list.append(sorted(nbs))
        return nb_list


def _get_nb(nb, idx):
    """Get neighbours for index idx from either array or list."""
    return nb[idx]


def _verify_coverage(codewords, q, n, radius, words, nb, strides):
    """Check if codewords cover all words. Returns (covered_all, uncovered_count)."""
    total = words.shape[0]
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, strides))
        nbs = _get_nb(nb, idx)
        covered[nbs] = True
    return covered.all(), int((~covered).sum())


def _greedy_cover(q, n, radius, words, nb, seed, time_limit, strides):
    """Greedy set cover with vectorized updates."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()

    covered = np.zeros(total, dtype=bool)
    codewords = []

    # Precompute: for each word, how many other words have it as a neighbour
    # i.e., the column sums of the neighbour matrix
    if isinstance(nb, np.ndarray):
        # nb is (total, nb_size) - compute column sums
        # Use sparse-like approach: count how many rows contain each column value
        # For efficiency, we'll track uncovered_count differently
        # uncovered_count[i] = number of uncovered words that are neighbours of i
        # Initially all words are uncovered, so uncovered_count[i] = nb_size for all i
        # When we cover a word c, we need to decrement uncovered_count for all i that have c as neighbour
        # This is the transpose of nb
        # For small total, just compute it
        uncovered_count = np.full(total, nb.shape[1], dtype=np.int32)
    else:
        uncovered_count = np.array([len(nb[i]) for i in range(total)], dtype=np.int32)

    while True:
        if time.time() - start > time_limit:
            break
        best_idx = int(np.argmax(uncovered_count))
        if uncovered_count[best_idx] <= 0:
            break
        codewords.append(words[best_idx].tolist())
        nbs = _get_nb(nb, best_idx)

        # Find which neighbours are newly covered
        newly_covered = nbs[~covered[nbs]]

        if len(newly_covered) > 0:
            covered[newly_covered] = True
            # For each newly covered word c, decrement uncovered_count for all words i that have c as neighbour
            # This requires knowing the reverse mapping
            # For speed, we'll just recompute uncovered_count from scratch periodically
            # Or: for each c in newly_covered, find all i such that c is in nb[i]
            # This is expensive. Instead, let's use a different approach:
            # uncovered_count[i] = number of j in nb[i] such that not covered[j]
            # We can update by: for each c in newly_covered, for each i in reverse_nb[c], uncovered_count[i] -= 1

        # Recompute uncovered_count efficiently:
        # For each i, uncovered_count[i] = sum of (not covered[nb[i]])
        # This is O(total * nb_size) per iteration which is too slow
        # Better: maintain it incrementally

        # Actually, let's just recompute using numpy:
        if isinstance(nb, np.ndarray):
            # uncovered_count[i] = number of neighbours of i that are not covered
            # = nb_size - number of neighbours of i that are covered
            # covered_count[i] = sum of covered[nb[i]]
            # This is a matrix-vector product: nb (as bool) @ covered
            # But nb is int, so: use np.bincount or direct
            # For total=2187, nb_size=16: this is 2187*16 = 35k ops, fine
            covered_nb = covered[nb]  # (total, nb_size) bool
            covered_count = covered_nb.sum(axis=1)  # (total,)
            uncovered_count = nb.shape[1] - covered_count
        else:
            # For list-based nb, recompute
            for i in range(total):
                cnt = 0
                for c in nb[i]:
                    if not covered[c]:
                        cnt += 1
                uncovered_count[i] = cnt

    return codewords, covered


def _local_improve(codewords, q, n, radius, words, nb, seed, time_limit, strides):
    """Remove redundant codewords using coverage counts."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    codewords = list(codewords)

    # Build word_to_idx
    word_to_idx = {}
    for i, w in enumerate(words):
        word_to_idx[tuple(w)] = i

    # Compute coverage count: for each word, how many codewords cover it
    cov_count = np.zeros(total, dtype=np.int32)
    cw_indices = []
    for cw in codewords:
        idx = word_to_idx[tuple(cw)]
        cw_indices.append(idx)
        nbs = _get_nb(nb, idx)
        cov_count[nbs] += 1

    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing each codeword in reverse order
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            idx = cw_indices[i]
            nbs = _get_nb(nb, idx)
            # Check if any word is only covered by this codeword
            if np.any(cov_count[nbs] == 1):
                continue
            # Safe to remove
            cov_count[nbs] -= 1
            codewords.pop(i)
            cw_indices.pop(i)
            improved = True

    return codewords


def _swap_improve(codewords, q, n, radius, words, nb, seed, time_limit, strides):
    """Try to reduce codeword count by finding words that can replace multiple codewords."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    codewords = list(codewords)

    word_to_idx = {}
    for i, w in enumerate(words):
        word_to_idx[tuple(w)] = i

    # Compute coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    cw_indices = []
    for cw in codewords:
        idx = word_to_idx[tuple(cw)]
        cw_indices.append(idx)
        nbs = _get_nb(nb, idx)
        cov_count[nbs] += 1

    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing redundant codewords
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            idx = cw_indices[i]
            nbs = _get_nb(nb, idx)
            if np.any(cov_count[nbs] == 1):
                continue
            cov_count[nbs] -= 1
            codewords.pop(i)
            cw_indices.pop(i)
            improved = True

        if time.time() - start > time_limit:
            break

        # Try 1-for-2 swaps: find a word w that covers all words covered by two specific codewords
        # that have no unique coverage after removing them
        if len(codewords) >= 2:
            # Sample a few pairs
            n_trials = min(50, (len(codewords) * (len(codewords) - 1) // 2))
            for _ in range(n_trials):
                if time.time() - start > time_limit:
                    break
                i1 = rng.randint(0, len(codewords) - 1)
                i2 = rng.randint(0, len(codewords) - 1)
                if i1 == i2:
                    continue
                idx1 = cw_indices[i1]
                idx2 = cw_indices[i2]
                nbs1 = _get_nb(nb, idx1)
                nbs2 = _get_nb(nb, idx2)

                # Check if removing both leaves all words covered
                # A word is uncovered if cov_count[word] <= 1 and word is in nbs1 or nbs2
                # and cov_count[word] counts only these two
                # More precisely: after removing both, a word c is uncovered if:
                #   cov_count[c] <= 2 and c is in (nbs1 ∪ nbs2) and
                #   (cov_count[c] == 1 and c is in nbs1 or c is in nbs2) or
                #   (cov_count[c] == 2 and c is in both nbs1 and nbs2)
                # Actually simpler: temporarily decrement and check
                union_nbs = np.unique(np.concatenate([nbs1, nbs2]))
                # Check: for all c in union_nbs, cov_count[c] >= 2 OR c is not uniquely covered by these two
                # A word c is uniquely covered by {i1,i2} if cov_count[c] == 1 and c in nbs1∪nbs2,
                # or cov_count[c] == 2 and c in nbs1∩nbs2
                # Actually: after removing both, c is uncovered iff cov_count[c] <= 2 and
                # c is in nbs1 or nbs2, and the number of {i1,i2} covering c equals cov_count[c]
                # i.e., cov_count[c] - (1 if c in nbs1 else 0) - (1 if c in nbs2 else 0) <= 0

                in_nbs1 = np.isin(union_nbs, nbs1)
                in_nbs2 = np.isin(union_nbs, nbs2)
                remaining = cov_count[union_nbs] - in_nbs1.astype(np.int32) - in_nbs2.astype(np.int32)
                if np.any(remaining <= 0):
                    continue

                # Find a word w that covers all of union_nbs
                # w must be such that union_nbs ⊆ nb[w]
                # i.e., for all c in union_nbs, w is in reverse_nb[c]
                # Or: nb[w] contains all of union_nbs
                # This is expensive to check for all w. Instead, find candidate w from union_nbs
                # that maximizes overlap with union_nbs
                best_w = -1
                best_overlap = 0
                for c in union_nbs[:20]:  # sample
                    if time.time() - start > time_limit:
                        break
                    nbs_c = _get_nb(nb, int(c))
                    overlap = np.isin(union_nbs, nbs_c).sum()
                    if overlap == len(union_nbs):
                        best_w = int(c)
                        best_overlap = overlap
                        break
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_w = int(c)

                if best_w >= 0 and best_overlap == len(union_nbs):
                    # Replace i1 and i2 with best_w
                    # Remove the larger index first
                    if i1 > i2:
                        i1, i2 = i2, i1
                    # Remove i2 (higher index)
                    cov_count[nbs2] -= 1
                    codewords.pop(i2)
                    cw_indices.pop(i2)
                    # Adjust i1 if needed
                    if i1 > i2:
                        i1 -= 1
                    # Remove i1
                    nbs1 = _get_nb(nb, cw_indices[i1])
                    cov_count[nbs1] -= 1
                    codewords.pop(i1)
                    cw_indices.pop(i1)
                    # Add best_w
                    nbs_w = _get_nb(nb, best_w)
                    cov_count[nbs_w] += 1
                    codewords.append(words[best_w].tolist())
                    cw_indices.append(best_w)
                    improved = True

    return codewords


def construct(q, n, radius):
    start = time.time()

    # Precompute words and neighbours
    words = _all_words(q, n)
    nb = _build_neighbours(q, n, radius, words)
    total = words.shape[0]
    strides = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    best_codewords = None
    best_count = float('inf')

    # Multiple greedy runs with different seeds
    seeds = [42, 123, 456, 789, 1011, 2022, 3141, 5678, 9012, 13579]
    n_seeds = len(seeds)

    # Time budget: use most of the deadline
    time_budget_total = _DEADLINE - time.time() - 3  # 3s safety margin
    time_per_greedy = time_budget_total * 0.5 / n_seeds
    time_per_improve = time_budget_total * 0.5 / n_seeds

    for si, seed in enumerate(seeds):
        elapsed = time.time() - start
        if elapsed > _DEADLINE - 5:
            break
        try:
            # Greedy cover
            t_start = time.time()
            codewords, covered = _greedy_cover(q, n, radius, words, nb, seed, time_per_greedy, strides)

            # Verify coverage
            cov_all, unc = _verify_coverage(codewords, q, n, radius, words, nb, strides)
            if not cov_all:
                # Fallback: add uncovered words
                uncovered = np.where(~covered)[0]
                for idx in uncovered:
                    codewords.append(words[idx].tolist())
                    covered[_get_nb(nb, idx)] = True

            # Local improvement: removals then swaps
            if time.time() < _DEADLINE - 5:
                improve_time = min(time_per_improve, _DEADLINE - 5 - time.time())
                if improve_time > 0.5:
                    improved = _local_improve(codewords, q, n, radius, words, nb, seed, improve_time * 0.4, strides)
                    # Then try swap improvement
                    if time.time() < _DEADLINE - 5:
                        swap_time = min(improve_time * 0.6, _DEADLINE - 5 - time.time())
                        if swap_time > 0.5:
                            improved = _swap_improve(improved, q, n, radius, words, nb, seed, swap_time, strides)
                    cov_all2, unc2 = _verify_coverage(improved, q, n, radius, words, nb, strides)
                    if cov_all2:
                        codewords = improved

            if len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords[:]

        except Exception:
            continue

    if best_codewords is not None:
        return best_codewords

    # Fallback: direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(3), repeat=n)]
# EVOLVE-BLOCK-END
