import time
import math
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Strong initial candidates
    candidates = [
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 1, 5, 7],
        [0, 1, 2, 7],
        [0, 1, 2, 6],
        [0, 1, 3, 8],
        [0, 1, 4, 9]
    ]
    
    best_U = candidates[0]
    best_score = compute_score(best_U)
    
    for c in candidates:
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_U = c[:]
            
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline

    # Targeted Grid Search for Size 4
    # Focus on small values where ratio is high
    for a in range(1, 10):
        if time.time() > deadline: break
        for b in range(a + 1, 20):
            if time.time() > deadline: break
            for c in range(b + 1, 30):
                if not try_set([0, a, b, c]):
                    break

    # Local Search
    current_U = best_U[:]
    current_score = best_score
    max_val = 50
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        U_set = set(current_U)
        
        # Try adding
        for new_elem in range(1, max_val + 1):
            if new_elem in U_set:
                continue
            candidate = sorted(list(U_set) + [new_elem])
            s = compute_score(candidate)
            if s > current_score:
                current_score = s
                current_U = candidate
                U_set = set(current_U)
                if s > best_score:
                    best_score = s
                    best_U = current_U[:]
                improved = True
                break
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
            
        # Try removing
        for i in range(1, len(current_U)):
            candidate = current_U[:i] + current_U[i+1:]
            if len(candidate) < 2:
                continue
            s = compute_score(candidate)
            if s > current_score:
                current_score = s
                current_U = candidate
                U_set = set(current_U)
                if s > best_score:
                    best_score = s
                    best_U = current_U[:]
                improved = True
                break
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
            
        # Try replacing
        for i in range(1, len(current_U)):
            old_elem = current_U[i]
            for new_elem in range(1, max_val + 1):
                if new_elem == old_elem or new_elem in U_set:
                    continue
                candidate = current_U[:]
                candidate[i] = new_elem
                candidate = sorted(candidate)
                s = compute_score(candidate)
                if s > current_score:
                    current_score = s
                    current_U = candidate
                    U_set = set(current_U)
                    if s > best_score:
                        best_score = s
                        best_U = current_U[:]
                    improved = True
                    break
                if time.time() > deadline - 5:
                    break
            if improved:
                break
            if time.time() > deadline - 5:
                break

    return best_U
