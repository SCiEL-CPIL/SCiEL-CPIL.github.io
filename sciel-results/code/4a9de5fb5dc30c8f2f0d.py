import time
import math
import numpy as np

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    n_sum = len(sumset)
    n_diff = len(diffset)
    
    if n_sum == 0:
        return -1
    
    ratio = n_diff / n_sum
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 1.0
    
    score = 1 + log_ratio / log_denom
    return score

def construct():
    deadline = time.time() + 110  # 110s safety margin
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Try geometric progressions with various ratios
    for n in range(2, 20):
        for r in range(2, 30):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                U.append(val)
                val *= r
            max_u = max(U)
            if max_u > 1000000:
                break
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_set = list(U)
    
    # Try to improve by local search: add or remove elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        U = list(best_set)
        max_u = max(U)
        
        # Try adding elements
        for candidate in range(1, min(max_u + 50, 5000)):
            if candidate not in U:
                new_U = sorted(U + [candidate])
                score = compute_score(new_U)
                if score > best_score + 1e-9:
                    best_score = score
                    best_set = new_U
                    improved = True
                    break
        
        # Try removing elements (except 0)
        for i in range(1, len(U)):
            new_U = U[:i] + U[i+1:]
            if len(new_U) >= 2:
                score = compute_score(new_U)
                if score > best_score + 1e-9:
                    best_score = score
                    best_set = new_U
                    improved = True
                    break
    
    return best_set
