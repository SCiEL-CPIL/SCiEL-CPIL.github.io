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
    
    # Optimization: iterate over list
    U_list = list(U)
    for a in U_list:
        for b in U_list:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        # If ratio <= 1, log is <= 0, score <= 1.
        # We can return early if we only care about > 1? 
        # But SA needs accurate scores for worse moves too.
        pass
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
        
    score = 1 + log_ratio / log_denom
    return score

def construct():
    # Parameters
    M = 100  # Bounded maximum for the binary vector representation
    time_limit = 110 # Leave 10s for safety
    start_time = time.time()
    
    best_set = [0]
    best_score = 1.0 # Score of {0} is technically undefined (0/0 or log(1/1)=0 -> 1), but max_u=0 returns 0.0 in my code. 
    # Let's fix score_set for {0}. If U={0}, max_u=0. return 0.0.
    # We need a valid set. {0,1} -> score 1.0.
    
    # Initialize with a few diverse candidates to ensure we start > 1.0 if possible
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 4, 10],
        [0, 2, 3, 7],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    current_set = list(best_set)
    current_score = best_score
    
    # Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    cool_rate = 0.999
    
    # We will run multiple restarts
    num_restarts = 10
    iterations_per_restart = 20000
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Reset or perturb current state
        if r == 0:
            current_set = list(best_set)
            current_score = best_score
        else:
            # Random restart with some bias towards best
            if random.random() < 0.5:
                current_set = list(best_set)
                # Perturb
                if len(current_set) > 1:
                    # Remove a random non-zero
                    idx = random.choice([x for x in current_set if x != 0])
                    current_set.remove(idx)
                # Add a random
                v = random.randint(1, M)
                if v not in current_set:
                    current_set.append(v)
                    current_set.sort()
            else:
                # Pure random
                size = random.randint(2, 15)
                others = random.sample(range(1, M + 1), size - 1)
                current_set = sorted([0] + others)
            
            current_score = score_set(current_set)
            
        T = T_start
        no_improve = 0
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Mutation: Flip a random bit in [1, M]
            v = random.randint(1, M)
            
            if v in current_set:
                # Remove
                new_set = [x for x in current_set if x != v]
            else:
                # Add
                new_set = current_set + [v]
                new_set.sort()
                
            # Ensure 0 is present (it always is, as we don't flip 0)
            
            new_score = score_set(new_set)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            # Cool down
            T *= cool_rate
            if T < T_end:
                T = T_end
                
            # Early break if stuck for too long in this restart
            if no_improve > 5000:
                break
                
    return best_set
