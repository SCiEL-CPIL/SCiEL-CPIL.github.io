import math
import time
import random

def calculate_score(U):
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_U = max(U)
    if max_U == 0:
        return 1.0
    
    # Compute sumset and diffset
    sum_set = set()
    diff_set = set()
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0.0
        
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_U + 1)
    if denom == 0:
        return 1.0
        
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 1.0
    best_U = [0, 1, 3]
    
    # Pre-computed good candidates
    candidates = [
        [0, 1, 3, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 1, 3, 4, 6], # Just in case
    ]
    
    for U in candidates:
        s = calculate_score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Heuristic search for better sets
    # Try to improve best_U or find new ones
    # We will try random subsets for M in range(1, 25)
    
    for M in range(1, 25):
        if time.time() > deadline - 10.0:
            break
            
        # Try some specific structures
        # 1. Random subsets
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            # Pick random size
            n = random.randint(3, 8)
            # Pick random elements
            elems = [0] + random.sample(range(1, M + 1), min(n - 1, M))
            U = sorted(elems)
            s = calculate_score(U)
            if s > best_score:
                best_score = s
                best_U = U
                
        # 2. Greedy construction
        # Start with 0, add elements one by one to maximize local score
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            U = [0]
            # Random start
            if random.random() < 0.5:
                U.append(random.randint(1, M))
            
            for _ in range(10): # Max 10 elements
                if time.time() > deadline - 2.0:
                    break
                best_add = -1
                best_local_score = calculate_score(U)
                
                for x in range(1, M + 1):
                    if x in U:
                        continue
                    U_new = U + [x]
                    s_new = calculate_score(U_new)
                    if s_new > best_local_score:
                        best_local_score = s_new
                        best_add = x
                
                if best_add == -1:
                    break
                U.append(best_add)
                U.sort()
                
                if best_local_score > best_score:
                    best_score = best_local_score
                    best_U = U

    return best_U
