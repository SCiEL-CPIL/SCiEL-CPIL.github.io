import time
import random
import math
import numpy as np

def compute_metrics(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    A is assumed to be a list of distinct positive integers.
    """
    n = len(A)
    A_arr = np.array(A, dtype=np.int64)
    
    # Compute all i <= j pairs
    i_idx, j_idx = np.triu_indices(n)
    sums = A_arr[i_idx] + A_arr[j_idx]
    prods = A_arr[i_idx] * A_arr[j_idx]
    
    sum_size = len(np.unique(sums))
    prod_size = len(np.unique(prods))
    
    return max(sum_size, prod_size)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Candidate pool
    max_pool_val = 2000
    base_pool = list(range(1, max_pool_val + 1))
    
    # Generate structured initial candidates
    initial_candidates = []
    
    # 1. Small consecutive sets (good sum collisions)
    initial_candidates.append(list(range(1, 21)))
    initial_candidates.append(list(range(2, 22)))
    initial_candidates.append(list(range(1, 22)))
    
    # 2. Multiples of small numbers (good product collisions)
    for base in [2, 3, 4, 5, 6]:
        cand = [base * i for i in range(1, 21) if base * i <= 10000]
        if len(cand) >= 20:
            initial_candidates.append(cand[:20])
    
    # 3. Powers and mixed
    initial_candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 1, 2, 3, 5, 7, 11, 13])
    initial_candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 36, 40, 48, 60, 72])
    
    # 4. Highly composite-like numbers
    hcn = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
    initial_candidates.append(hcn)
    
    # 5. Random sets from small range
    for _ in range(30):
        cand = random.sample(range(1, 101), 20)
        initial_candidates.append(cand)
        
    # 6. Random sets from medium range
    for _ in range(15):
        cand = random.sample(range(1, 501), 20)
        initial_candidates.append(cand)

    # Validate and sort initial candidates
    valid_initials = []
    for cand in initial_candidates:
        if len(cand) == 20 and len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            valid_initials.append(sorted(cand))
            
    if not valid_initials:
        valid_initials = [sorted(random.sample(base_pool, 20))]
        
    # Evaluate initial candidates to find the best starting point
    best_set = None
    best_score = float('inf')
    
    for cand in valid_initials:
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
            
    # Multiple restarts with simulated annealing
    num_restarts = 3
    restart_deadline = deadline - 5.0  # Leave 5s for final polish
    time_per_restart = (restart_deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > restart_deadline:
            break
            
        # Start from best found so far or a random start
        if restart == 0:
            current_set = best_set[:]
        else:
            # Perturb best_set
            current_set = best_set[:]
            for _ in range(random.randint(3, 8)):
                idx = random.randint(0, 19)
                old_val = current_set[idx]
                new_val = random.choice(base_pool)
                attempts = 0
                while new_val in current_set and attempts < 20:
                    new_val = random.choice(base_pool)
                    attempts += 1
                if new_val not in current_set:
                    current_set[idx] = new_val
            current_set = sorted(current_set)
        
        current_score = compute_metrics(current_set)
        
        # Simulated Annealing
        T_start = 10.0
        T_end = 0.01
        restart_end = min(restart_deadline, start_time + time_per_restart * (restart + 1))
        max_steps = 50000
        
        num_steps = 0
        while time.time() < restart_end and num_steps < max_steps:
            step = num_steps
            num_steps += 1
            
            # Cooling schedule
            progress = min(1.0, num_steps / 20000.0)
            T = T_start * (T_end / T_start) ** progress
            
            # Random Move: Replace one element
            idx = random.randint(0, 19)
            old_val = current_set[idx]
            
            # Choose new value with bias towards smaller numbers
            if random.random() < 0.7:
                new_val = random.randint(1, 500)
            else:
                new_val = random.choice(base_pool)
            
            # Check duplicates
            if new_val in current_set:
                found = False
                for _ in range(10):
                    new_val = random.choice(base_pool)
                    if new_val not in current_set:
                        found = True
                        break
                if not found:
                    continue
            
            current_set[idx] = new_val
            new_score = compute_metrics(current_set)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta <= 0 or (T > 0 and random.random() < min(1.0, -delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_set[:]
            else:
                current_set[idx] = old_val
            
            # Occasionally do a swap
            if num_steps % 5 == 0 and time.time() < restart_end:
                 idx1 = random.randint(0, 19)
                 idx2 = random.randint(0, 19)
                 if idx1 != idx2:
                     v1 = current_set[idx1]
                     v2 = current_set[idx2]
                     new_v1 = random.choice(base_pool)
                     new_v2 = random.choice(base_pool)
                     
                     # Ensure distinctness
                     if new_v1 == new_v2:
                         continue
                     if new_v1 in current_set or new_v2 in current_set:
                         continue
                         
                     current_set[idx1] = new_v1
                     current_set[idx2] = new_v2
                     new_score = compute_metrics(current_set)
                     
                     delta = new_score - current_score
                     if delta <= 0 or (T > 0 and random.random() < min(1.0, -delta / T)):
                         current_score = new_score
                         if new_score < best_score:
                             best_score = new_score
                             best_set = current_set[:]
                     else:
                         current_set[idx1] = v1
                         current_set[idx2] = v2

    # Final Polish: Hill Climbing on best_set
    if time.time() < deadline - 1.0:
        improved = True
        polish_deadline = min(deadline - 0.5, time.time() + 8.0)
        
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                # Try a focused pool for polish
                polish_pool = list(range(1, 301))
                for new_val in polish_pool:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val
                
    return sorted(best_set)
