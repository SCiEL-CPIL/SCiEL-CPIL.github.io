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

def _score_set_fast_v2(U):
    """
    Even faster scoring: exploit symmetry. U+U is symmetric around 0 in the sense
    that if we sort U, the sumset size can be computed from unique pairs.
    For U-U, |U-U| = 2*|{a-b: a>b in U}| + 1 (since 0 is always in U-U).
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
    
    if n <= 1:
        return 1.0
    
    # For sumset: use the fact that a+b = b+a, so we only need i<=j pairs
    # But numpy's outer + gives all n*n, so let's just use unique on the full matrix
    # Actually for n up to ~50, n^2 = 2500 which is fine
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # For diffset: a-b where a>b gives positive diffs, a<b gives negative
    # |U-U| = 2 * |{a-b : a,b in U, a>b}| + 1
    # We can compute just the upper triangle
    if n > 1:
        # Upper triangle: i > j means U[i] > U[j] (since sorted)
        # diffs where row > col
        i_idx, j_idx = np.triu_indices(n, k=1)
        pos_diffs = U_arr[i_idx] - U_arr[j_idx]
        pos_diff_unique = len(np.unique(pos_diffs))
        diffset_size = 2 * pos_diff_unique + 1
    else:
        diffset_size = 1
    
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

def _build_golomb_ruler(n):
    """
    Build a Golomb-ruler-like set: choose elements so that all pairwise
    differences are distinct (or nearly distinct). This maximizes |U-U|.
    Use greedy: at each step, pick the smallest positive integer that
    doesn't create a duplicate difference.
    """
    if n <= 1:
        return [0]
    U = [0]
    diffs = set()
    for _ in range(1, n):
        candidate = 1
        while True:
            # Check if adding candidate creates a duplicate difference
            new_diffs = set()
            ok = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    ok = False
                    break
                new_diffs.add(d)
                # Also check negative (u - candidate)
                nd = u - candidate
                if nd in diffs:
                    ok = False
                    break
                new_diffs.add(nd)
            if ok:
                U.append(candidate)
                diffs.update(new_diffs)
                break
            candidate += 1
    return sorted(U)

def _build_sparse_golomb(n, max_val=500):
    """
    Sparse Golomb ruler: allow some duplicate differences but keep them sparse.
    Greedy with a penalty for duplicate differences.
    """
    if n <= 1:
        return [0]
    U = [0]
    diff_counts = {}
    for _ in range(1, n):
        best_candidate = None
        best_penalty = float('inf')
        for candidate in range(1, max_val + 1):
            # Count how many new duplicate differences this would create
            penalty = 0
            for u in U:
                d = candidate - u
                nd = u - candidate
                if d in diff_counts:
                    penalty += 1
                if nd in diff_counts:
                    penalty += 1
            if penalty < best_penalty:
                best_penalty = penalty
                best_candidate = candidate
                if penalty == 0:
                    break
        if best_candidate is None:
            break
        U.append(best_candidate)
        for u in U[:-1]:
            d = best_candidate - u
            nd = u - best_candidate
            diff_counts[d] = diff_counts.get(d, 0) + 1
            diff_counts[nd] = diff_counts.get(nd, 0) + 1
    return sorted(U)

def _local_search(U, max_iterations=500):
    """
    Local search: try adding/removing elements to improve score.
    Uses the faster v2 scorer.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = list(set(U) | {0})
        U.sort()
    
    best_score = _score_set_fast_v2(U)
    best_U = list(U)
    
    max_u = max(U) if U else 0
    # Search space: elements up to some limit
    search_max = max(3 * max_u, 200)
    
    for it in range(max_iterations):
        if _time_remaining() < 5:
            break
        
        improved = False
        # Try adding an element - focus on candidates that are "sparse"
        # relative to existing elements
        candidates_to_try = []
        for candidate in range(1, search_max + 1):
            if candidate in U:
                continue
            # Quick heuristic: prefer candidates far from existing elements
            # (creates new differences without many new sums)
            min_dist = min(abs(candidate - u) for u in U if u != candidate)
            candidates_to_try.append((min_dist, candidate))
        
        # Try the most "distant" candidates first
        candidates_to_try.sort(reverse=True)
        
        for min_dist, candidate in candidates_to_try[:30]:  # Limit to top 30
            new_U = list(U) + [candidate]
            new_score = _score_set_fast_v2(new_U)
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
                new_score = _score_set_fast_v2(new_U)
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
    best_score = _score_set_fast_v2([0])
    
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
    
    # Golomb ruler constructions
    for n in range(3, 12):
        initializations.append(_build_golomb_ruler(n))
    
    # Sparse Golomb constructions
    for n in range(3, 12):
        initializations.append(_build_sparse_golomb(n, max_val=300))
    
    # Random initializations with Golomb-like properties
    random.seed(42)
    for _ in range(30):
        n = random.randint(3, 12)
        U = [0]
        used_diffs = set()
        for _ in range(1, n):
            # Try to pick a candidate that creates few duplicate diffs
            best_cand = None
            best_pen = float('inf')
            for _attempt in range(50):
                cand = random.randint(1, 500)
                if cand in U:
                    continue
                pen = 0
                for u in U:
                    d = cand - u
                    if d in used_diffs:
                        pen += 1
                if pen < best_pen:
                    best_pen = pen
                    best_cand = cand
                    if pen == 0:
                        break
            if best_cand is not None:
                U.append(best_cand)
                for u in U[:-1]:
                    d = best_cand - u
                    used_diffs.add(d)
        initializations.append(sorted(U))
    
    # Evaluate and locally optimize each
    for init_U in initializations:
        if _time_remaining() < 10:
            break
        
        # Quick score to filter
        quick_score = _score_set_fast_v2(init_U)
        if quick_score < best_score - 0.3:
            continue
        
        optimized_U, optimized_score = _local_search(init_U, max_iterations=150)
        if optimized_score > best_score:
            best_score = optimized_score
            best_U = optimized_U
    
    return best_U, best_score

def _exhaustive_small(max_u, min_size=3, max_size=8):
    """
    Exhaustive search for small sets with max element <= max_u.
    """
    best_U = [0]
    best_score = _score_set_fast_v2([0])
    
    candidates = list(range(0, max_u + 1))
    
    for size in range(min_size, max_size + 1):
        if _time_remaining() < 10:
            break
        
        for combo in combinations(candidates, size):
            if 0 not in combo:
                continue
            U = list(combo)
            score = _score_set_fast_v2(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast_v2(best_U)
    
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
