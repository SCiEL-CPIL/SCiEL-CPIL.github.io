import time
import math
import numpy as np
import random

def compute_score(U):
    """Compute the score for a given set U using vectorized operations."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    # Compute sumset U+U using broadcasting
    sums = U_arr[:, None] + U_arr[None, :]
    n_sum = len(np.unique(sums))
    
    # Compute difference set U-U using broadcasting
    diffs = U_arr[:, None] - U_arr[None, :]
    n_diff = len(np.unique(diffs))
    
    if n_sum == 0:
        return -1
    
    ratio = n_diff / n_sum
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 1.0
    
    score = 1 + log_ratio / log_denom
    return score

def generate_candidates(U, max_val):
    """Generate candidate elements based on structure of U."""
    U_set = set(U)
    candidates = set()
    
    # Sums of pairs
    for a in U:
        for b in U:
            s = a + b
            if 0 < s <= max_val and s not in U_set:
                candidates.add(s)
    
    # Differences (positive only, since we need nonnegative)
    for a in U:
        for b in U:
            d = a - b
            if 0 < d <= max_val and d not in U_set:
                candidates.add(d)
    
    # Small integers not in U
    for i in range(1, min(50, max_val + 1)):
        if i not in U_set:
            candidates.add(i)
    
    return list(candidates)

def local_search_dynamic(U, deadline, best_score):
    """Local search with dynamic candidate generation."""
    U = list(U)
    best_score = max(best_score, compute_score(U))
    
    while time.time() < deadline:
        max_u = max(U)
        # Limit max_u to keep computation tractable
        max_u_limit = min(max_u + 200, 50000)
        
        candidates = generate_candidates(U, max_u_limit)
        random.shuffle(candidates)
        
        improved = False
        for cand in candidates:
            if time.time() > deadline:
                break
            new_U = sorted(U + [cand])
            score = compute_score(new_U)
            if score > best_score + 1e-10:
                best_score = score
                U = new_U
                improved = True
                break
        
        if not improved:
            # Try removing elements
            removed = False
            for i in range(1, len(U)):
                if time.time() > deadline:
                    break
                new_U = U[:i] + U[i+1:]
                if len(new_U) >= 2:
                    score = compute_score(new_U)
                    if score > best_score + 1e-10:
                        best_score = score
                        U = new_U
                        removed = True
                        break
            if not removed:
                break
    
    return U, best_score

def construct():
    deadline = time.time() + 110  # 110s safety margin
    
    # Start with several initial candidates
    initial_sets = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 9],
        [0, 1, 2, 5, 10],
        [0, 1, 4, 16, 64],
        [0, 2, 4, 8, 16],
    ]
    
    best_set = initial_sets[0]
    best_score = compute_score(best_set)
    
    # Try geometric progressions
    for n in range(2, 15):
        for r in range(2, 20):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                U.append(val)
                val *= r
            if max(U) > 100000:
                break
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_set = list(U)
    
    # Run local search from best initial set
    best_set, best_score = local_search_dynamic(best_set, deadline, best_score)
    
    # Try local search from other initial sets if time permits
    for init in initial_sets[1:]:
        if time.time() > deadline:
            break
        s = compute_score(init)
        if s >= best_score - 0.01:  # Only try if close
            candidate_set, candidate_score = local_search_dynamic(init, deadline, best_score)
            if candidate_score > best_score:
                best_score = candidate_score
                best_set = candidate_set
    
    return best_set
