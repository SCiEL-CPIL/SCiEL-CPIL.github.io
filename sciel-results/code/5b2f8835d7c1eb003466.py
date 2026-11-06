import random
import time
import math

def _score_set(S):
    S_set = set(S)
    covered_diffs = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            d = b - a
            c = 2 * b - a
            if c in S_set:
                covered_diffs.add(d)
    return len(covered_diffs)

def construct():
    start_time = time.time()
    time_limit = 115
    
    N = 32
    MAX_VAL = 512
    
    best_score = 0
    best_set = set()
    
    def gen_random_set():
        return set(random.sample(range(1, MAX_VAL + 1), N))
    
    def simulated_annealing(initial_set, max_iters):
        current_set = set(initial_set)
        current_score = _score_set(current_set)
        
        T_start = 5.0
        T_end = 0.1
        if max_iters > 0:
            alpha = (T_end / T_start) ** (1.0 / max_iters)
        else:
            alpha = 1.0
            
        T = T_start
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Move: remove one, add one
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            
            to_add = random.randint(1, MAX_VAL)
            while to_add in current_set:
                to_add = random.randint(1, MAX_VAL)
            
            current_set.add(to_add)
            
            new_score = _score_set(current_set)
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
            else:
                if delta < 0 and T > 0:
                    # Avoid division by zero or underflow issues
                    if T > 0.01:
                        prob = math.exp(delta / T)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_set.remove(to_add)
                        current_set.add(to_remove)
                else:
                    # delta == 0, accept to move around
                    # or reject? Usually accept to explore.
                    pass 
            
            T *= alpha
            
        return current_set, current_score

    # Phase 1: Multiple restarts
    while time.time() - start_time < time_limit - 20:
        initial = gen_random_set()
        s, score = simulated_annealing(initial, 5000)
        
        if score > best_score:
            best_score = score
            best_set = s

    # Phase 2: Polish best
    if best_set:
        s, score = simulated_annealing(best_set, 50000)
        if score > best_score:
            best_score = score
            best_set = s

    return list(best_set)
