import time
import math
import random
import numpy as np
from itertools import combinations, product

# Deadline guard: evaluator kills after 150s, we target 120s hard limit with margin
_START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leave 5s margin for overhead

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed()

def _score_set(U):
    """
    Compute the lower bound: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    U must be a finite set of nonnegative integers containing 0.
    """
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all a+b for a,b in U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Difference set: all a-b for a,b in U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return 0.0
    
    log_ratio = math.log(diff_size / sum_size)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """
    Faster scoring using numpy for larger sets.
    """
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    # Sumset via outer sum
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Difference set via outer diff
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
    
    log_ratio = math.log(diffset_size / sumset_size)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_denom

def _build_geometric(ratio, n_terms):
    """Build a geometric progression set."""
    if ratio <= 1:
        return [0]
    U = [0]
    current = 1
    for i in range(1, n_terms):
        U.append(current)
        current = int(current * ratio)
    return U

def _build_binary(n_bits):
    """Build a set based on binary representation (powers of 2)."""
    return [0] + [1 << i for i in range(n_bits)]

def _build_mixed(base, step, n_terms):
    """Build a mixed arithmetic-geometric set."""
    U = [0]
    current = base
    for i in range(1, n_terms):
        U.append(current)
        current = current + step
    return U

def _local_search(U, max_iterations=500):
    """
    Local search: try adding/removing elements to improve score.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = list(set(U) | {0})
        U.sort()
    
    best_score = _score_set_fast(U)
    best_U = list(U)
    
    max_u = max(U) if U else 0
    # Search space: elements up to some limit
    search_max = max(2 * max_u, 100)
    
    for it in range(max_iterations):
        if _time_remaining() < 5:
            break
        
        improved = False
        # Try adding an element
        for candidate in range(1, search_max + 1):
            if candidate in U:
                continue
            new_U = list(U) + [candidate]
            new_score = _score_set_fast(new_U)
            if new_score > best_score + 1e-12:
                U = new_U
                best_score = new_score
                best_U = list(U)
                improved = True
                search_max = max(search_max, 2 * max(U))
                break
        
        # Try removing an element (not 0)
        if not improved:
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                new_U = U[:i] + U[i+1:]
                if not new_U:
                    continue
                new_score = _score_set_fast(new_U)
                if new_score > best_score + 1e-12:
                    U = new_U
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_U, best_score

def _multi_phase_search():
    """
    Multi-phase local search with different initializations.
    """
    best_U = [0]
    best_score = _score_set_fast([0])
    
    # Phase 1: Try various structured initializations
    initializations = []
    
    # Geometric progressions
    for ratio in [1.5, 2, 2.5, 3, 3.5, 4, 5]:
        for n_terms in range(3, 15):
            initializations.append(_build_geometric(ratio, n_terms))
    
    # Binary sets
    for n_bits in range(2, 12):
        initializations.append(_build_binary(n_bits))
    
    # Mixed arithmetic-geometric
    for base in [1, 2, 3]:
        for step in [1, 2, 3, 5]:
            for n_terms in range(3, 12):
                initializations.append(_build_mixed(base, step, n_terms))
    
    # Simple arithmetic progressions
    for step in [1, 2, 3, 5, 7, 10]:
        for n_terms in range(3, 15):
            U = [0] + [i * step for i in range(1, n_terms)]
            initializations.append(U)
    
    # Random initializations
    random.seed(42)
    for _ in range(20):
        n = random.randint(3, 10)
        max_val = random.randint(10, 100)
        U = [0] + random.sample(range(1, max_val + 1), min(n-1, max_val))
        initializations.append(sorted(U))
    
    # Evaluate and locally optimize each
    for init_U in initializations:
        if _time_remaining() < 10:
            break
        
        # Quick score to filter
        quick_score = _score_set_fast(init_U)
        if quick_score < best_score - 0.5:
            continue
        
        optimized_U, optimized_score = _local_search(init_U, max_iterations=200)
        if optimized_score > best_score:
            best_score = optimized_score
            best_U = optimized_U
    
    return best_U, best_score

def _exhaustive_small(max_u, min_size=3, max_size=8):
    """
    Exhaustive search for small sets with max element <= max_u.
    """
    best_U = [0]
    best_score = _score_set_fast([0])
    
    candidates = list(range(0, max_u + 1))
    
    for size in range(min_size, max_size + 1):
        if _time_remaining() < 10:
            break
        
        for combo in combinations(candidates, size):
            if 0 not in combo:
                continue
            U = list(combo)
            score = _score_set_fast(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Exhaustive search for small configurations
    try:
        small_U, small_score = _exhaustive_small(max_u=15, min_size=3, max_size=8)
        if small_score > best_score:
            best_score = small_score
            best_U = small_U
    except Exception:
        pass
    
    # Phase 2: Multi-phase local search with structured candidates
    try:
        multi_U, multi_score = _multi_phase_search()
        if multi_score > best_score:
            best_score = multi_score
            best_U = multi_U
    except Exception:
        pass
    
    # Phase 3: Further local search from the best found so far
    try:
        if _time_remaining() > 10:
            final_U, final_score = _local_search(best_U, max_iterations=1000)
            if final_score > best_score:
                best_score = final_score
                best_U = final_U
    except Exception:
        pass
    
    # Ensure 0 is in the set and return sorted list
    best_U = sorted(set(best_U) | {0})
    
    return best_U
