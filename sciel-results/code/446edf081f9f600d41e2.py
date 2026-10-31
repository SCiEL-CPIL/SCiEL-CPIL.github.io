import time
import math
import random

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
            
    if len(sum_set) == 0:
        return 0.0
        
    log_num = math.log(len(diff_set) / len(sum_set))
    log_den = math.log(2 * max_U + 1)
    
    if log_den <= 0:
        return 0.0
    return 1.0 + log_num / log_den

def _local_search(U, time_limit):
    """Perform local search with swap, add, and remove moves."""
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    best_score = _compute_score(U)
    best_U = U[:]
    
    deadline = time.time() + time_limit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best_U)
        max_val = max(best_U)
        
        # Determine search range for new elements
        # We don't want to go too far, as it increases the denominator
        search_max = max_val * 2 + 20
        
        # 1. Try Swaps (replace one element with another)
        # Generate candidate values not in best_U
        candidates = [x for x in range(1, search_max + 1) if x not in best_U]
        random.shuffle(candidates)
        
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = best_U[i]
            # Try replacing best_U[i] with each candidate
            # We limit the number of candidates tried per element to avoid getting stuck
            # in a slow loop if no improvement is found
            limit = min(len(candidates), 50) 
            for new_val in candidates[:limit]:
                if time.time() > deadline:
                    break
                candidate_U = best_U[:]
                candidate_U[i] = new_val
                candidate_U = list(set(candidate_U))
                if 0 not in candidate_U:
                    continue
                s = _compute_score(candidate_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate_U
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue

        # 2. Try Adding an element (increase size)
        if time.time() < deadline and n < 12: # Cap size to prevent explosion
            add_candidates = [x for x in range(1, max_val + 10) if x not in best_U]
            random.shuffle(add_candidates)
            limit = min(len(add_candidates), 50)
            for new_val in add_candidates[:limit]:
                if time.time() > deadline:
                    break
                candidate_U = best_U + [new_val]
                s = _compute_score(candidate_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate_U
                    improved = True
                    break
            if improved:
                continue

        # 3. Try Removing an element (decrease size)
        if time.time() < deadline and n > 3:
            # Try removing non-zero elements
            indices_to_try = [i for i in range(n) if best_U[i] != 0]
            random.shuffle(indices_to_try)
            for i in indices_to_try:
                if time.time() > deadline:
                    break
                candidate_U = best_U[:]
                candidate_U.pop(i)
                s = _compute_score(candidate_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate_U
                    improved = True
                    break
            if improved:
                continue

    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    total_deadline = _START_TIME + 110.0  # Safety margin
    
    # Optimal Golomb Rulers (or close to it)
    # N=3: 0, 1, 3
    # N=4: 0, 1, 4, 6
    # N=5: 0, 1, 4, 9, 11
    # N=6: 0, 1, 4, 10, 12, 17
    # N=7: 0, 1, 4, 10, 12, 17, 20
    # N=8: 0, 1, 4, 10, 12, 17, 20, 23 (Approx)
    
    initial_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 23],
    ]
    
    # Add some random sets with small max values to explore different structures
    random.seed(42)
    for _ in range(20):
        n = random.randint(3, 8)
        # Sample distinct values
        vals = random.sample(range(1, 40), n-1)
        vals.sort()
        initial_sets.append([0] + vals)
        
    best_overall = initial_sets[0]
    best_score_overall = _compute_score(best_overall)
    
    # Shuffle initial sets to prioritize diverse starts
    random.shuffle(initial_sets)
    
    for U in initial_sets:
        if time.time() > total_deadline - 5.0:
            break
        
        time_left = total_deadline - time.time() - 2.0
        if time_left < 1.0:
            continue
        
        U_ls, score_ls = _local_search(U, time_left)
        
        if score_ls > best_score_overall:
            best_score_overall = score_ls
            best_overall = U_ls[:]
            
    return best_overall
