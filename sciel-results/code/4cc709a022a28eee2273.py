"""Rewritten program for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in _product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _product(iterable, repeat):
    """Simple itertools.product replacement to avoid import."""
    if repeat == 0:
        yield ()
        return
    pool = tuple(iterable)
    for prefix in _product(pool, repeat - 1):
        for x in pool:
            yield prefix + (x,)


def _build_cov_matrix(q, n, radius):
    """Build the full coverage matrix: cov_matrix[i] is a bool array of words within radius of word i."""
    space_size = q ** n
    cov = np.zeros((space_size, space_size), dtype=bool)
    digits = np.zeros((space_size, n), dtype=np.int32)
    for i in range(space_size):
        x = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = x % q
            x //= q
    cov[np.arange(space_size), np.arange(space_size)] = True
    if radius >= 1:
        for pos in range(n):
            for d in range(q):
                mask = digits[:, pos] != d
                if not np.any(mask):
                    continue
                src_indices = np.where(mask)[0]
                weight = q ** (n - 1 - pos)
                targets = src_indices + (d - digits[src_indices, pos]) * weight
                cov[src_indices, targets] = True
    return cov, digits


def _word_to_idx(q, word):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(q, n, idx):
    word = []
    for pos in range(n - 1, -1, -1):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _local_search(q, n, radius, words, time_limit=120.0):
    """Local search: iteratively try to remove codewords, and swap to improve coverage."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    
    word_indices = np.array([_word_to_idx(q, w) for w in words], dtype=np.int64)
    k = len(word_indices)
    
    # coverage_count[p] = number of selected codewords covering position p
    sel_cov = cov[word_indices]
    coverage_count = sel_cov.sum(axis=0).astype(np.int32)
    
    improved = True
    while improved:
        if time.time() - _start_time > time_limit - 5:
            break
        improved = False
        
        # Try to remove each codeword
        for i in range(len(word_indices)):
            if time.time() - _start_time > time_limit - 5:
                break
            idx_i = word_indices[i]
            # Check if any position is only covered by this codeword
            only_covered = coverage_count == 1
            if np.any(sel_cov[i] & only_covered):
                continue
            # Safe to remove
            coverage_count -= sel_cov[i]
            word_indices = np.delete(word_indices, i)
            sel_cov = np.delete(sel_cov, i, axis=0)
            improved = True
            break  # restart loop after removal
    
    # Convert back to words
    return [_idx_to_word(q, n, int(idx)) for idx in word_indices]


def _greedy_cover_fast(q, n, radius, time_limit=120.0):
    """Fast greedy set cover using precomputed neighbor lists."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    
    uncovered_count = cov.sum(axis=1).astype(np.int32)
    uncovered = np.ones(space_size, dtype=bool)
    
    selected = []
    remaining = space_size
    
    while remaining > 0:
        if time.time() - _start_time > time_limit - 5:
            break
        best = int(np.argmax(uncovered_count))
        if uncovered_count[best] <= 0:
            break
        selected.append(best)
        
        newly_covered_mask = cov[best] & uncovered
        if not np.any(newly_covered_mask):
            break
        nc_indices = np.where(newly_covered_mask)[0]
        uncovered_count -= cov[:, nc_indices].sum(axis=1).astype(np.int32)
        uncovered_count = np.maximum(uncovered_count, 0)
        
        uncovered[newly_covered_mask] = False
        remaining -= len(nc_indices)
        uncovered_count[best] = -1
    
    return [_idx_to_word(q, n, i) for i in selected]


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Start with tetracode-based solution (729 words)
    seed_words = _tetracode_solution(q, n, radius)
    
    # Local search to prune
    pruned = _local_search(q, n, radius, seed_words, time_limit)
    
    # If time remains, try greedy from scratch
    elapsed = time.time() - _start_time
    if elapsed < time_limit - 30 and len(pruned) > 500:
        greedy_fresh = _greedy_cover_fast(q, n, radius, time_limit=time_limit)
        if len(greedy_fresh) < len(pruned):
            pruned = greedy_fresh
        pruned = _local_search(q, n, radius, pruned, time_limit)
    
    return pruned
# EVOLVE-BLOCK-END
