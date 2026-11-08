import time
import random
import math
import numpy as np
from itertools import combinations

def score_set(U):
    """Fast scoring using numpy for sum and diff sets"""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    
    arr = np.array(U, dtype=np.int64)
    n = len(arr)
    
    # Compute all sums using outer product
    sums = np.add.outer(arr, arr)
    sum_set = set(sums.flatten().tolist())
    
    # Compute all differences
    diffs = np.subtract.outer(arr, arr)
    diff_set = set(diffs.flatten().tolist())
    
    sum_count = len(sum_set)
    diff_count = len(diff_set)
    
    if sum_count == 0:
        return -1.0
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return 1.0
    
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return 1.0
    
    return 1 + math.log(ratio) / math.log(D)

def exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets to find good seeds"""
    best_set = None
    best_score = -1.0
    
    if n == 3:
        for a in range(1, min(max_val + 1, 50)):
            for b in range(a + 1, min(a * 3 + 10, max_val + 1)):
                if time.time() > deadline:
                    return best_set, best_score
                candidate = [0, a, b]
                s = score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
    elif n == 4:
        for a in range(1, min(max_val + 1, 30)):
            for b in range(a + 1, min(a * 2 + 15, max_val + 1)):
                for c in range(b + 1, min(b * 2 + 20, max_val + 1)):
                    if time.time() > deadline:
                        return best_set, best_score
                    candidate = [0, a, b, c]
                    s = score_set(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
    elif n == 5:
        for a in range(1, min(max_val + 1, 20)):
            for b in range(a + 1, min(a * 2 + 10, max_val + 1)):
                for c in range(b + 1, min(b * 2 + 15, max_val + 1)):
                    for d in range(c + 1, min(c * 2 + 20, max_val + 1)):
                        if time.time() > deadline:
                            return best_set, best_score
                        candidate = [0, a, b, c, d]
                        s = score_set(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
    
    return best_set, best_score

def local_search_dynamic(U, deadline, max_iter=100):
    """Local search with dynamic candidate generation"""
    best_set = list(U)
    best_score = score_set(U)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        n = len(best_set)
        max_u = max(best_set)
        
        # Try adding a new element
        candidates_to_try = []
        
        # Small additions near max
        for delta in range(1, min(30, max_u // 2 + 1)):
            candidates_to_try.append(max_u + delta)
        
        # Geometric additions
        for ratio in [1.5, 2, 2.5, 3, 4]:
            candidates_to_try.append(int(max_u * ratio))
            candidates_to_try.append(int(max_u * ratio) + 1)
        
        # Random additions
        for _ in range(10):
            candidates_to_try.append(random.randint(max_u + 1, max_u * 3 + 50))
        
        # Try removing an element (if n > 3)
        if n > 3:
            for i in range(1, n):
                candidate = best_set[:i] + best_set[i+1:]
                s = score_set(candidate)
                if s > best_score + 1e-10:
                    best_set = candidate
                    best_score = s
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try adding candidates
        best_candidate = None
        best_candidate_score = best_score
        
        for new_val in set(candidates_to_try):
            if new_val <= 0 or new_val in best_set:
                continue
            candidate = sorted(best_set + [new_val])
            s = score_set(candidate)
            if s > best_candidate_score + 1e-10:
                best_candidate_score = s
                best_candidate = candidate
        
        if best_candidate is not None:
            best_set = best_candidate
            best_score = best_candidate_score
        else:
            # Try swapping: remove one, add another
            if n >= 3:
                swap_improved = False
                for remove_idx in range(1, n):
                    if time.time() > deadline - 5:
                        break
                    base = best_set[:remove_idx] + best_set[remove_idx+1:]
                    base_max = max(base) if base else 0
                    
                    for new_val in range(base_max + 1, base_max + 20):
                        if new_val in base:
                            continue
                        candidate = sorted(base + [new_val])
                        s = score_set(candidate)
                        if s > best_score + 1e-10:
                            best_set = candidate
                            best_score = s
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                
                if not swap_improved:
                    break
            else:
                break
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Start with a good seed
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Phase 1: Exhaustive search for small sets to find better seeds
    for n in [3, 4, 5]:
        found_set, found_score = exhaustive_search_small(n, 200, deadline - 30)
        if found_set is not None and found_score > best_score:
            best_set = found_set
            best_score = found_score
    
    # Phase 2: Local search from the best seed
    final_set, final_score = local_search_dynamic(best_set, deadline, max_iter=50)
    if final_score > best_score:
        best_set = final_set
        best_score = final_score
    
    # Phase 3: Try different initial seeds and local search
    alternative_seeds = [
        [0, 1, 2],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 5],
        [0, 2, 3, 7],
        [0, 1, 3, 8],
    ]
    
    for seed in alternative_seeds:
        if time.time() > deadline - 10:
            break
        s = score_set(seed)
        if s < best_score - 0.05:
            continue
        result_set, result_score = local_search_dynamic(seed, deadline - 5, max_iter=30)
        if result_score > best_score:
            best_set = result_set
            best_score = result_score
    
    return best_set
