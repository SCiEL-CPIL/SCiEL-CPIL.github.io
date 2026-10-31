import time
import random
import math
from itertools import combinations

_START_TIME = None

def _compute_score(U):
    """Compute the score for set U."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    max_U = max(U_set)
    if max_U == 0:
        return 0.0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    log_num = math.log(len(diff_set) / len(sum_set))
    log_den = math.log(2 * max_U + 1)
    if log_den <= 0:
        return 0.0
    return 1.0 + log_num / log_den

def _local_search(U, time_limit):
    """Perform local search with single and double swap moves."""
    global _START_TIME
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    best_score = _compute_score(U)
    best_U = U[:]
    
    deadline = _START_TIME + time_limit
    
    # Determine search range
    max_val = max(U)
    search_max = max_val * 2 + 100
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        current = best_U[:]
        current_score = best_score
        
        # Single swap: replace one element with another
        for i in range(len(current)):
            if time.time() > deadline - 2.0:
                break
            old_val = current[i]
            # Try removing and adding a new value
            for new_val in range(1, search_max + 1):
                if new_val == old_val:
                    continue
                candidate = current[:]
                candidate[i] = new_val
                candidate = list(set(candidate))
                if 0 not in candidate:
                    continue
                if len(candidate) != len(current):
                    continue  # Skip if we lost an element (duplicate)
                s = _compute_score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate[:]
                    improved = True
                    current = candidate[:]
                    current_score = s
                    break
        
        # Double swap: replace two elements
        if time.time() > deadline - 2.0:
            break
        n = len(best_U)
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    break
                old1, old2 = best_U[i], best_U[j]
                # Try replacing with two new values
                for new_val1 in range(1, search_max + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if new_val1 == old1 or new_val1 == old2:
                        continue
                    for new_val2 in range(1, search_max + 1):
                        if new_val2 == old1 or new_val2 == old2 or new_val2 == new_val1:
                            continue
                        candidate = best_U[:]
                        candidate[i] = new_val1
                        candidate[j] = new_val2
                        candidate = list(set(candidate))
                        if 0 not in candidate:
                            continue
                        if len(candidate) != n:
                            continue
                        s = _compute_score(candidate)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_U = candidate[:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_U, best_score

def _greedy_construct(target_size, max_val, seed):
    """Greedy construction: add elements that maximize marginal score gain."""
    random.seed(seed)
    U = {0}
    # Start with a small base
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    # Add a few random elements to start
    for _ in range(min(3, target_size - 1)):
        U.add(random.randint(1, max_val))
    
    current_score = _compute_score(list(U))
    
    for _ in range(target_size - len(U)):
        best_gain = -float('inf')
        best_elem = None
        # Sample candidates
        sample_size = min(200, len(candidates))
        sample = random.sample(candidates, sample_size)
        for elem in sample:
            if elem in U:
                continue
            U.add(elem)
            s = _compute_score(list(U))
            U.remove(elem)
            gain = s - current_score
            if gain > best_gain:
                best_gain = gain
                best_elem = elem
        if best_elem is not None:
            U.add(best_elem)
            current_score = _compute_score(list(U))
        else:
            break
    
    return list(U), current_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    total_deadline = _START_TIME + 110.0  # Safety margin
    
    best_overall = [0, 1, 3]
    best_score_overall = _compute_score([0, 1, 3])
    
    # Try multiple strategies with different parameters
    strategies = [
        # (target_size, max_val, seed)
        (5, 50, 42),
        (6, 80, 123),
        (7, 120, 456),
        (8, 150, 789),
        (5, 30, 1000),
        (6, 40, 2000),
    ]
    
    for target_size, max_val, seed in strategies:
        if time.time() > total_deadline - 30.0:
            break
        
        # Greedy construction
        U, score = _greedy_construct(target_size, max_val, seed)
        
        # Local search
        time_left = total_deadline - time.time() - 5.0
        if time_left < 5.0:
            continue
        
        U_ls, score_ls = _local_search(U, time_left)
        
        if score_ls > best_score_overall:
            best_score_overall = score_ls
            best_overall = U_ls[:]
    
    # Final polish: try more aggressive local search on best
    if time.time() < total_deadline - 10.0:
        time_left = total_deadline - time.time() - 5.0
        if time_left > 5.0:
            U_final, score_final = _local_search(best_overall, time_left)
            if score_final > best_score_overall:
                best_score_overall = score_final
                best_overall = U_final[:]
    
    return best_overall
