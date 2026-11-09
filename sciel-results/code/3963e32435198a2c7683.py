# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random
import math


def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    return bool(np.all(dist.min(axis=1) <= radius))


def _words_from_indices(indices, q, n):
    result = []
    for idx in indices:
        word = []
        rem = int(idx)
        for _ in range(n):
            word.append(rem % q)
            rem //= q
        result.append(word)
    return result


def _build_neighbors(q, n, radius):
    """neighbors[i] = list of word-indices within Hamming distance <= radius of word i."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    neighbors = []
    # For radius 1 we can build fast; otherwise fall back to distance matrix for small spaces.
    if radius == 1:
        for i in range(total):
            w = all_words[i]
            nbrs = [i]
            for pos in range(n):
                orig = w[pos]
                for v in range(q):
                    if v != orig:
                        w[pos] = v
                        nbrs.append(int(np.ravel_multi_index(tuple(w), (q,) * n)))
                w[pos] = orig
            neighbors.append(nbrs)
    else:
        diff = (all_words[:, None, :] != all_words[None, :, :])
        dist = diff.sum(axis=2)
        for i in range(total):
            neighbors.append(list(np.where(dist[i] <= radius)[0]))
    return neighbors


def _remove_redundant(code_indices, neighbors, deadline):
    """Greedy removal of codewords whose ball is fully covered by others."""
    total = len(neighbors)
    coverage = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        for nb in neighbors[ci]:
            coverage[nb] += 1
    idx_set = set(code_indices)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for pos in range(len(code_indices)):
            ci = code_indices[pos]
            # is every neighbor covered by at least one other codeword?
            redundant = True
            for nb in neighbors[ci]:
                if coverage[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                for nb in neighbors[ci]:
                    coverage[nb] -= 1
                code_indices.pop(pos)
                idx_set.discard(ci)
                changed = True
                break
    return code_indices, coverage


def _simulated_annealing(code_indices, neighbors, q, n, radius, deadline, rng):
    """SA refinement: remove/add/swap moves; minimize (uncovered, size)."""
    total = len(neighbors)
    coverage = np.zeros(total, dtype=np.int32)
    idx_set = set(code_indices)
    for ci in code_indices:
        for nb in neighbors[ci]:
            coverage[nb] += 1
    uncovered = int(np.count_nonzero(coverage == 0))

    best_code = list(code_indices)
    best_uncovered = uncovered
    best_size = len(code_indices)

    T = 2.0
    T_min = 0.005
    cooling = 0.9995
    e = 2.718281828459045

    while time.time() < deadline and T > T_min:
        r = rng.random()
        if r < 0.45 and len(code_indices) > 1:
            # Remove
            pos = rng.randrange(len(code_indices))
            ci = code_indices[pos]
            delta_un = 0
            for nb in neighbors[ci]:
                if coverage[nb] == 1:
                    delta_un += 1
            for nb in neighbors[ci]:
                coverage[nb] -= 1
            code_indices.pop(pos)
            idx_set.discard(ci)
            uncovered += delta_un
            size = len(code_indices)
            if uncovered < best_uncovered or (uncovered == best_uncovered and size < best_size):
                best_uncovered, best_size, best_code = uncovered, size, list(code_indices)
            elif uncovered > best_uncovered:
                if rng.random() > math.exp(-(uncovered - best_uncovered) / T):
                    # revert
                    for nb in neighbors[ci]:
                        coverage[nb] += 1
                    code_indices.insert(pos, ci)
                    idx_set.add(ci)
                    uncovered -= delta_un
        elif r < 0.75:
            # Add
            new_ci = rng.randrange(total)
            if new_ci in idx_set:
                continue
            delta_un = 0
            for nb in neighbors[new_ci]:
                if coverage[nb] == 0:
                    delta_un -= 1
            for nb in neighbors[new_ci]:
                coverage[nb] += 1
            code_indices.append(new_ci)
            idx_set.add(new_ci)
            uncovered += delta_un
            size = len(code_indices)
            if uncovered < best_uncovered or (uncovered == best_uncovered and size < best_size):
                best_uncovered, best_size, best_code = uncovered, size, list(code_indices)
            elif size > best_size:
                if rng.random() > math.exp(-(size - best_size) / (T * 5.0)):
                    for nb in neighbors[new_ci]:
                        coverage[nb] -= 1
                    code_indices.pop()
                    idx_set.discard(new_ci)
                    uncovered -= delta_un
        else:
            # Swap
            if len(code_indices) < 1:
                continue
            pos = rng.randrange(len(code_indices))
            ci = code_indices[pos]
            new_ci = rng.randrange(total)
            if new_ci in idx_set:
                continue
            # combined delta
            delta_un = 0
            for nb in neighbors[ci]:
                if coverage[nb] == 1 and (coverage[nb] - 1) + (1 if nb in neighbors[new_ci] else 0) == 0:
                    # after removing ci, is nb still covered by new_ci or others?
                    pass
            # simpler: apply then measure
            for nb in neighbors[ci]:
                coverage[nb] -= 1
            for nb in neighbors[new_ci]:
                coverage[nb] += 1
            code_indices[pos] = new_ci
            idx_set.discard(ci)
            idx_set.add(new_ci)
            new_uncovered = int(np.count_nonzero(coverage == 0))
            size = len(code_indices)
            if new_uncovered < best_uncovered or (new_uncovered == best_uncovered and size < best_size):
                best_uncovered, best_size, best_code = new_uncovered, size, list(code_indices)
                uncovered = new_uncovered
            elif new_uncovered > best_uncovered:
                if rng.random() > math.exp(-(new_uncovered - best_uncovered) / T):
                    uncovered = new_uncovered
                else:
                    # revert
                    for nb in neighbors[new_ci]:
                        coverage[nb] -= 1
                    for nb in neighbors[ci]:
                        coverage[nb] += 1
                    code_indices[pos] = ci
                    idx_set.discard(new_ci)
                    idx_set.add(ci)
        T *= cooling
    return best_code


def _greedy_cover_v2(q, n, radius, deadline):
    """Improved greedy with multiple strategies and polish."""
    if n == 0:
        return [[]]

    M = q ** n
    if M > 200000:
        return _base_cover(q, n, radius)

    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    diff = (words[:, None, :] != words[None, :, :])
    dist = diff.sum(axis=2)
    del diff
    ball = dist <= radius
    del dist
    del words

    col_sums = ball.sum(axis=1).astype(np.int64)

    best_code = None
    best_size = M + 1

    # Strategy 1: Standard greedy (max gain)
    covered = np.zeros(M, dtype=bool)
    code_idx = []
    remaining = M
    while remaining > 0 and time.time() < deadline:
        cand = ~covered
        gains = np.where(cand, col_sums, -1)
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            j = int(np.argmax(cand))
        code_idx.append(j)
        newcov = ball[j] & ~covered
        if newcov.any():
            covered |= newcov
            remaining -= int(newcov.sum())
    if remaining == 0:
        cand_code = _words_from_indices(code_idx, q, n)
        if len(cand_code) < best_size:
            best_size = len(cand_code)
            best_code = cand_code

    # Strategy 2: Greedy with tie-breaking by minimum overlap
    if time.time() < deadline - 5:
        covered = np.zeros(M, dtype=bool)
        code_idx = []
        remaining = M
        chosen_mask = np.zeros(M, dtype=bool)
        while remaining > 0 and time.time() < deadline:
            cand = ~covered
            gains = np.where(cand, col_sums, -1)
            max_gain = gains.max()
            if max_gain <= 0:
                j = int(np.argmax(cand))
            else:
                max_cands = np.where((gains == max_gain) & cand)[0]
                if len(max_cands) == 1:
                    j = int(max_cands[0])
                else:
                    overlaps = ball[max_cands][:, chosen_mask].sum(axis=1)
                    j = int(max_cands[np.argmin(overlaps)])
            code_idx.append(j)
            chosen_mask[j] = True
            newcov = ball[j] & ~covered
            if newcov.any():
                covered |= newcov
                remaining -= int(newcov.sum())
        if remaining == 0:
            cand_code = _words_from_indices(code_idx, q, n)
            if len(cand_code) < best_size:
                best_size = len(cand_code)
                best_code = cand_code

    # Strategy 3: Random restarts
    rng = random.Random(12345)
    for trial in range(3):
        if time.time() > deadline - 3:
            break
        covered = np.zeros(M, dtype=bool)
        code_idx = []
        remaining = M
        while remaining > 0 and time.time() < deadline:
            cand = ~covered
            gains = np.where(cand, col_sums, -1)
            max_gain = gains.max()
            if max_gain <= 0:
                cand_indices = np.where(cand)[0]
                j = int(cand_indices[rng.randrange(len(cand_indices))])
            else:
                max_cands = np.where((gains == max_gain) & cand)[0]
                j = int(max_cands[rng.randrange(len(max_cands))])
            code_idx.append(j)
            newcov = ball[j] & ~covered
            if newcov.any():
                covered |= newcov
                remaining -= int(newcov.sum())
        if remaining == 0:
            cand_code = _words_from_indices(code_idx, q, n)
            if len(cand_code) < best_size:
                best_size = len(cand_code)
                best_code = cand_code

    if best_code is None:
        return _base_cover(q, n, radius)
    return best_code


def _local_polish(codewords, q, n, radius, deadline):
    """Remove redundant codewords via coverage matrix."""
    if not codewords:
        return codewords
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    diff = (all_words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    cover_matrix = dist <= radius
    current_code = list(range(len(C)))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in current_code:
            if time.time() > deadline:
                break
            others = [c for c in current_code if c != i]
            if np.all(cover_matrix[:, others].any(axis=1)):
                current_code.remove(i)
                changed = True
                break
    return [list(C[i]) for i in current_code]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0

    if n == 0:
        return [[]]

    base = _base_cover(q, n, radius)
    base = base[:1000000]
    if not _is_cover(base, q, n, radius):
        base = [list(w) for w in itertools.product(range(q), repeat=n)]
    best = base

    # Greedy multi-strategy
    try:
        cand = _greedy_cover_v2(q, n, radius, deadline - 40.0)
        if len(cand) < len(best) and _is_cover(cand, q, n, radius):
            best = cand
    except Exception:
        pass

    # Simulated annealing refinement (the key added machinery)
    try:
        if time.time() < deadline - 25.0:
            M = q ** n
            if M <= 200000:
                neighbors = _build_neighbors(q, n, radius)
                all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
                word_to_idx = {}
                for i, w in enumerate(all_words):
                    word_to_idx[tuple(w)] = i
                start_idx = [word_to_idx[tuple(cw)] for cw in best]
                # Pre-remove redundants
                start_idx, _ = _remove_redundant(start_idx, neighbors, deadline - 15.0)
                rng = random.Random(7)
                sa_code = _simulated_annealing(start_idx, neighbors, q, n, radius, deadline - 10.0, rng)
                sa_words = _words_from_indices(sa_code, q, n)
                if _is_cover(sa_words, q, n, radius) and len(sa_words) < len(best):
                    best = sa_words
    except Exception:
        pass

    # Final local polish
    try:
        if time.time() < deadline - 3.0:
            polished = _local_polish(best, q, n, radius, deadline - 1.0)
            if _is_cover(polished, q, n, radius) and len(polished) < len(best):
                best = polished
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
