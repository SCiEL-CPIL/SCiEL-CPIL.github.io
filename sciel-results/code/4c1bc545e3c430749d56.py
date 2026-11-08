import time
import math
import numpy as np
from itertools import combinations

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    U_arr = np.array(U, dtype=np.int64)
    sum_set = np.unique(U_arr[:, None] + U_arr[None, :])
    diff_set = np.unique(U_arr[:, None] - U_arr[None, :])
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
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

    # Phase 1: Brute force search for small sets
    # Search for sets {0, a, b, c} with 0 < a < b < c <= 50
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            if time.time() > deadline:
                break
            for c in range(b + 1, 51):
                if not try_set([0, a, b, c]):
                    break

    # Search for sets {0, a, b, c, d} with 0 < a < b < c < d <= 30
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 15):
            if time.time() > deadline:
                break
            for c in range(b + 1, 22):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 31):
                    if not try_set([0, a, b, c, d]):
                        break

    # Search for sets {0, a, b, c, d, e} with 0 < a < b < c < d < e <= 25
    for a in range(1, 6):
        if time.time() > deadline:
            break
        for b in range(a + 1, 10):
            if time.time() > deadline:
                break
            for c in range(b + 1, 15):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 20):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 26):
                        if not try_set([0, a, b, c, d, e]):
                            break

    # Phase 2: Local search with multiple restarts
    if time.time() < deadline - 10:
        # Try multiple random restarts
        import random
        random.seed(42)
        
        restarts = 0
        max_restarts = 50
        
        while time.time() < deadline - 5 and restarts < max_restarts:
            restarts += 1
            
            # Start from current best or a random set
            if restarts == 1:
                current_U = best_U[:]
            else:
                # Random set of size 4-7
                size = random.randint(4, 7)
                candidates = list(range(1, 60))
                random.shuffle(candidates)
                current_U = [0] + sorted(candidates[:size-1])
            
            # Local search iterations
            improved = True
            max_iter = 50
            iter_count = 0
            
            while improved and time.time() < deadline - 2 and iter_count < max_iter:
                improved = False
                iter_count += 1
                U_set = set(current_U)
                
                # Try adding a new element
                for new_elem in range(1, 80):
                    if new_elem in U_set:
                        continue
                    candidate = sorted(list(U_set) + [new_elem])
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        current_U = candidate
                        U_set = set(current_U)
                        improved = True
                        break
                    if time.time() > deadline - 2:
                        break
                
                if time.time() > deadline - 2:
                    break
                
                # Try removing an element (keep 0)
                for i in range(1, len(current_U)):
                    candidate = current_U[:i] + current_U[i+1:]
                    if len(candidate) < 2:
                        continue
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        current_U = candidate
                        U_set = set(current_U)
                        improved = True
                        break
                    if time.time() > deadline - 2:
                        break
                
                if time.time() > deadline - 2:
                    break
                
                # Try replacing an element (keep 0)
                for i in range(1, len(current_U)):
                    old_elem = current_U[i]
                    for new_elem in range(1, 80):
                        if new_elem == old_elem or new_elem in U_set:
                            continue
                        candidate = current_U[:]
                        candidate[i] = new_elem
                        candidate = sorted(candidate)
                        s = compute_score(candidate)
                        if s > best_score:
                            best_score = s
                            best_U = candidate
                            current_U = candidate
                            U_set = set(current_U)
                            improved = True
                            break
                        if time.time() > deadline - 2:
                            break
                    if improved:
                        break
                    if time.time() > deadline - 2:
                        break
                
                # Try 2-swap: replace two elements
                if not improved and len(current_U) >= 3 and time.time() < deadline - 2:
                    for i in range(1, len(current_U)):
                        for j in range(i+1, len(current_U)):
                            old_i = current_U[i]
                            old_j = current_U[j]
                            for new_i in range(1, 80):
                                if new_i == old_i or new_i in U_set:
                                    continue
                                for new_j in range(1, 80):
                                    if new_j == old_j or new_j in U_set or new_j == new_i:
                                        continue
                                    candidate = current_U[:]
                                    candidate[i] = new_i
                                    candidate[j] = new_j
                                    candidate = sorted(candidate)
                                    s = compute_score(candidate)
                                    if s > best_score:
                                        best_score = s
                                        best_U = candidate
                                        current_U = candidate
                                        U_set = set(current_U)
                                        improved = True
                                        break
                                    if time.time() > deadline - 2:
                                        break
                                if improved or time.time() > deadline - 2:
                                    break
                            if improved or time.time() > deadline - 2:
                                break
                        if improved or time.time() > deadline - 2:
                            break

    return best_U
