import random
import math
import time

def score_set(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Calculate U+U and U-U
    sum_set = set()
    diff_set = set()
    
    u_list = list(U)
    n = len(u_list)
    
    for i in range(n):
        a = u_list[i]
        for j in range(n):
            b = u_list[j]
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
        
    score = 1 + log_ratio / log_denom
    return score

def construct():
    # Parameters for Simulated Annealing over binary vectors (set membership)
    max_val = 100  # Upper bound for element values
    max_size = 15  # Max number of elements in the set
    start_temp = 0.5
    cool_rate = 0.995
    iterations = 60000
    start_time = time.time()
    time_limit = 110
    
    # Initial candidates: known good structures
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 6, 10],
        [0, 2, 3, 7, 8]
    ]
    
    best_set = candidates[0]
    best_score = score_set(best_set)
    
    # State is a set (binary vector representation)
    current_set = set(candidates[0])
    current_score = best_score
    
    temp = start_temp
    no_improve_count = 0
    restart_interval = 2000
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        new_set = set(current_set)
        r = random.random()
        
        if r < 0.4:
            # Single flip: Add or remove
            if len(new_set) < max_size:
                # Bias towards smaller values to keep max_val low
                if random.random() < 0.6:
                    candidate = random.randint(1, max_val // 2)
                else:
                    candidate = random.randint(1, max_val)
                
                if candidate not in new_set:
                    new_set.add(candidate)
            else:
                # Remove
                if len(new_set) > 1:
                    candidates_to_remove = [x for x in new_set if x != 0]
                    if candidates_to_remove:
                        to_remove = random.choice(candidates_to_remove)
                        new_set.discard(to_remove)
        
        elif r < 0.8:
            # Double flip: Swap
            if len(new_set) > 1:
                candidates_to_remove = [x for x in new_set if x != 0]
                if candidates_to_remove:
                    to_remove = random.choice(candidates_to_remove)
                    new_set.discard(to_remove)
                    
                    if random.random() < 0.6:
                        candidate = random.randint(1, max_val // 2)
                    else:
                        candidate = random.randint(1, max_val)
                    
                    if candidate not in new_set:
                        new_set.add(candidate)
        
        else:
            # Perturbation: Local move
            if len(new_set) > 1:
                candidates_to_change = [x for x in new_set if x != 0]
                if candidates_to_change:
                    to_change = random.choice(candidates_to_change)
                    delta = random.randint(-10, 10)
                    new_val = to_change + delta
                    if 0 < new_val <= max_val:
                        new_set.discard(to_change)
                        if new_val not in new_set:
                            new_set.add(new_val)
                        else:
                            new_set.add(to_change)
        
        if new_set == current_set:
            continue
            
        # Ensure 0 is present
        if 0 not in new_set:
            new_set.add(0)
            
        new_score = score_set(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        temp *= cool_rate
        if temp < 1e-6:
            temp = 1e-6
            
        if no_improve_count > restart_interval:
            if random.random() < 0.5:
                current_set = set(best_set)
            else:
                size = random.randint(3, max_size)
                others = random.sample(range(1, max_val + 1), size - 1)
                current_set = {0} | set(others)
            current_score = score_set(current_set)
            no_improve_count = 0
            temp = max(temp, 0.3)

    return best_set
