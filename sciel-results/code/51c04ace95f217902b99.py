import time
import math
import numpy as np
import random

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
    deadline = start_time + 145  # 5s safety margin
    
    best_U = [0, 1, 5, 7]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline

    # 1. Grid search for small sets
    # Size 4
    for a in range(1, 20):
        if time.time() > deadline: break
        for b in range(a + 1, 40):
            if time.time() > deadline: break
            for c in range(b + 1, 60):
                if not try_set([0, a, b, c]):
                    break

    # Size 5
    for a in range(1, 12):
        if time.time() > deadline: break
        for b in range(a + 1, 25):
            if time.time() > deadline: break
            for c in range(b + 1, 40):
                if time.time() > deadline: break
                for d in range(c + 1, 55):
                    if not try_set([0, a, b, c, d]):
                        break

    # Size 6 (coarser grid)
    for a in range(1, 8):
        if time.time() > deadline: break
        for b in range(a + 1, 18):
            if time.time() > deadline: break
            for c in range(b + 1, 30):
                if time.time() > deadline: break
                for d in range(c + 1, 45):
                    if time.time() > deadline: break
                    for e in range(d + 1, 60):
                        if not try_set([0, a, b, c, d, e]):
                            break

    # 2. Local Search with Simulated Annealing / Hill Climbing
    current_U = best_U[:]
    current_score = best_score
    
    # Pre-compute possible replacement values
    # We will search in range [1, 100]
    max_val = 100
    
    improved = True
    while improved and time.time() < deadline - 10:
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
                break # Restart loop with new state
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
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
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
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
                if time.time() > deadline - 10:
                    break
            if improved:
                break
            if time.time() > deadline - 10:
                break
        
        # Try swapping two elements (effectively replacing both)
        if len(current_U) >= 3 and time.time() < deadline - 10:
            for i in range(1, len(current_U)):
                for j in range(i + 1, len(current_U)):
                    # Try replacing u_i with new_i
                    for new_i in range(1, max_val + 1):
                        if new_i in U_set or new_i == current_U[i]:
                            continue
                        candidate = current_U[:]
                        candidate[i] = new_i
                        candidate = sorted(candidate)
                        if len(set(candidate)) < len(candidate):
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
                        if time.time() > deadline - 10:
                            break
                    if improved:
                        break
                if improved or time.time() > deadline - 10:
                    break

    # 3. Final Polish: Fine-tuning around best_U
    if time.time() < deadline - 2:
        final_U = best_U[:]
        final_set = set(final_U)
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(1, len(final_U)):
                old = final_U[i]
                # Try small deltas
                for delta in range(-5, 6):
                    new_val = old + delta
                    if new_val <= 0 or new_val in final_set:
                        continue
                    candidate = final_U[:]
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    if len(set(candidate)) < len(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        final_U = candidate
                        final_set = set(final_U)
                        improved = True
                        break
                    if time.time() > deadline - 1:
                        break
                if improved:
                    break
            if not improved:
                break

    return best_U
