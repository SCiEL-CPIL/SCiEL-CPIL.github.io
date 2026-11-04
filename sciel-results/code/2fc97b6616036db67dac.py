"""Optimized ternary covering code K_3(7,1) via greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

import numpy as np

# Precompute neighbors for all 2187 words
# neighbor[word_idx][pos] = list of neighbor indices
_NEIGHBORS = None
_Q = 3
_N = 7
_TOTAL = 3 ** 7  # 2187


def _ensure_neighbors():
    global _NEIGHBORS
    if _NEIGHBORS is not None:
        return
    _NEIGHBORS = []
    for w in range(_TOTAL):
        word = list(_idx_to_list(w))
        nbrs = [w]  # include self
        for pos in range(_N):
            for val in range(_Q):
                if word[pos] != val:
                    word[pos] = val
                    nbrs.append(_list_to_idx(word))
        word[pos] = val  # restore
        _NEIGHBORS.append(nbrs)


def _idx_to_list(idx):
    """Convert index to list of digits."""
    result = []
    for _ in range(_N):
        result.append(idx % _Q)
        idx //= _Q
    return result


def _list_to_idx(lst):
    """Convert list of digits to index."""
    result = 0
    for i in range(_N):
        result = result * _Q + lst[_N - 1 - i]
    return result


def _hamming_dist(a, b):
    """Hamming distance between two lists."""
    return sum(1 for i in range(_N) if a[i] != b[i])


def _greedy_cover():
    """Greedy set cover: repeatedly pick codeword covering most uncovered words."""
    _ensure_neighbors()
    uncovered = set(range(_TOTAL))
    codewords = []
    
    while uncovered:
        # Find codeword that covers the most uncovered neighbors
        best_word = None
        best_count = -1
        # Sample if too many to check all
        candidates = list(uncovered) if len(uncovered) <= 500 else random.sample(list(uncovered), 500)
        for w in candidates:
            count = sum(1 for nb in _NEIGHBORS[w] if nb in uncovered)
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            best_word = random.choice(list(uncovered))
        
        codewords.append(best_word)
        for nb in _NEIGHBORS[best_word]:
            uncovered.discard(nb)
    
    return codewords


def _is_covered(codewords):
    """Check if all words are within radius 1 of some codeword."""
    _ensure_neighbors()
    covered = np.zeros(_TOTAL, dtype=bool)
    for cw in codewords:
        for nb in _NEIGHBORS[cw]:
            covered[nb] = True
    return covered.all()


def _local_search(codewords, time_limit=60.0):
    """Local search: try removing codewords and swapping to improve coverage."""
    _ensure_neighbors()
    start = time.time()
    codewords = list(codewords)
    
    # Build coverage count: for each word, how many codewords cover it
    coverage = np.zeros(_TOTAL, dtype=np.int32)
    for cw in codewords:
        for nb in _NEIGHBORS[cw]:
            coverage[nb] += 1
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Try removing each codeword if all its covered words are still covered
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            cw = codewords[i]
            # Check if removing this codeword leaves any word uncovered
            removable = True
            for nb in _NEIGHBORS[cw]:
                if coverage[nb] == 1:
                    removable = False
                    break
            if removable:
                for nb in _NEIGHBORS[cw]:
                    coverage[nb] -= 1
                codewords.pop(i)
                improved = True
        
        # Try swapping: replace one codeword with another that covers more
        if (time.time() - start) > time_limit:
            break
        for i in range(len(codewords)):
            if time.time() - start > time_limit:
                break
            cw = codewords[i]
            # Try random alternative codewords
            for _ in range(20):
                alt = random.randint(0, _TOTAL - 1)
                if alt == cw:
                    continue
                # Compute net gain
                gain = 0
                for nb in _NEIGHBORS[alt]:
                    if coverage[nb] == 0:
                        gain += 1
                for nb in _NEIGHBORS[cw]:
                    if coverage[nb] == 1:
                        gain -= 1
                if gain > 0:
                    # Apply swap
                    for nb in _NEIGHBORS[cw]:
                        coverage[nb] -= 1
                    for nb in _NEIGHBORS[alt]:
                        coverage[nb] += 1
                    codewords[i] = alt
                    improved = True
                    break
    
    return codewords


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering radius radius."""
    global _Q, _N, _TOTAL, _NEIGHBORS
    _Q = q
    _N = n
    _TOTAL = q ** n
    _NEIGHBORS = None
    _ensure_neighbors()
    
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_code = None
    best_len = float('inf')
    
    # Multiple random restarts with greedy + local search
    num_restarts = 0
    while time.time() < deadline - 30.0:
        num_restarts += 1
        
        # Randomized greedy: shuffle the order of candidate selection
        random.seed(num_restarts * 7919)
        code = _greedy_cover()
        
        # Local search improvement
        code = _local_search(code, time_limit=min(40.0, deadline - time.time() - 10.0))
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = list(code)
        
        # If we've hit the known best (186) or better, try a bit more
        if best_len <= 186 and time.time() < deadline - 20.0:
            # Try more aggressive local search
            code = _local_search(best_code, time_limit=min(30.0, deadline - time.time() - 5.0))
            if len(code) < best_len:
                best_len = len(code)
                best_code = list(code)
    
    # Verify coverage before returning
    if best_code is None:
        best_code = _greedy_cover()
    
    # Final verification
    if not _is_covered(best_code):
        # Fallback: greedy without local search
        best_code = _greedy_cover()
    
    # Convert to list of lists
    result = [_idx_to_list(cw) for cw in best_code]
    
    # Ensure distinctness
    seen = set()
    unique = []
    for r in result:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    
    return unique
# EVOLVE-BLOCK-END
