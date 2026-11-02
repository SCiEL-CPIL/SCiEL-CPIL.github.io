import time
import math
import random

def _score_set(U):
    if not U or 0 not in U:
        return -1.0
    U = sorted(U)
    if len(U) < 2:
        return 1.0
        
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
            
    len_sums = len(sums)
    len_diffs = len(diffs)
    
    if len_sums == 0:
        return 1.0
        
    ratio = len_diffs / len_sums
    if ratio <= 1.0:
        # If ratio is 1 or less, score is <= 1. 
        # We want to maximize, so we prefer > 1.
        # But technically score is 1 + log(ratio)/log(...).
        # If ratio < 1, score < 1.
        pass
        
    max_u = max(U)
    denom_arg = 2 * max_u + 1
    if denom_arg <= 1:
        return 1.0
        
    num = math.log(ratio)
    den = math.log(denom_arg)
    
    if den == 0:
        return 1.0
        
    return 1.0 + num / den

def _local_search(initial_U, time_limit):
    current_U = sorted(list(initial_U))
    current_score = _score_set(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        # Strategy: 
        # 1. Modify an existing element
        # 2. Add a new element
        # 3. Remove an element
        
        move_type = random.random()
        
        if move_type < 0.6 and len(current_U) > 0:
            # Modify
            idx = random.randint(0, len(current_U) - 1)
            old_val = current_U[idx]
            # Try small changes
            change = random.randint(-5, 5)
            new_val = old_val + change
            if new_val < 0:
                continue
            if new_val in current_U and new_val != old_val:
                continue
            
            new_U = list(current_U)
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            
        elif move_type < 0.8 and len(current_U) < 20:
            # Add
            max_u = max(current_U)
            new_val = random.randint(0, max_u * 2 + 10)
            if new_val in current_U:
                continue
            new_U = list(current_U)
            new_U.append(new_val)
            new_U = sorted(new_U)
            
        elif len(current_U) > 3:
            # Remove
            idx = random.randint(0, len(current_U) - 1)
            if current_U[idx] == 0:
                continue # Cannot remove 0
            new_U = list(current_U)
            new_U.pop(idx)
            new_U = sorted(new_U)
        else:
            continue
            
        new_score = _score_set(new_U)
        
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
        else:
            no_improve_count += 1
            # Simple restart logic: if stuck for a while, jump
            if no_improve_count > 50:
                # Perturb current_U
                if len(current_U) > 0:
                    idx = random.randint(0, len(current_U) - 1)
                    if current_U[idx] != 0:
                        current_U[idx] = random.randint(0, max(current_U) * 2)
                        current_U = sorted(set(current_U))
                no_improve_count = 0
                current_score = _score_set(current_U)

    return best_U, best_score

def construct():
    deadline = time.time() + 140.0 # 10s margin
    best_global_U = [0, 1, 3]
    best_global_score = _score_set(best_global_U)
    
    # Try different sizes
    for N in range(3, 16):
        if time.time() > deadline:
            break
            
        # Initial guesses
        starts = []
        # Powers of 2
        starts.append([0] + [2**i for i in range(1, N)])
        # Random
        for _ in range(3):
            random_set = [0]
            for _ in range(N-1):
                random_set.append(random.randint(1, 50))
            random_set = sorted(set(random_set))
            if len(random_set) == N:
                starts.append(random_set)
        
        time_per_start = (deadline - time.time()) / (len(starts) * N)
        time_per_start = max(1.0, min(5.0, time_per_start))
        
        for start in starts:
            if time.time() > deadline:
                break
            # Ensure size is N
            while len(start) < N:
                start.append(random.randint(1, 100))
                start = sorted(set(start))
            while len(start) > N:
                # Remove non-zero
                idx = random.randint(1, len(start)-1)
                start.pop(idx)
                
            U, score = _local_search(start, time_per_start)
            if score > best_global_score:
                best_global_score = score
                best_global_U = U
                
    return best_global_U
