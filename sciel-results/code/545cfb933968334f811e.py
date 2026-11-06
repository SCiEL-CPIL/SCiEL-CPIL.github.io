import time
import random
import math

def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
    s_set = set(s)
    distinct_d = set()
    elems = sorted(s)
    n = len(elems)
    for i in range(n):
        for j in range(i + 1, n):
            d = elems[j] - elems[i]
            a2 = elems[j] + d
            if a2 in s_set:
                distinct_d.add(d)
    return len(distinct_d)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(777)
    
    # Structured initial candidates
    candidates = [
        list(range(1, 33)),
        [1 + 16 * i for i in range(32)],
        list(range(1, 65, 2)),
        [4 * i for i in range(1, 33)],
        random.sample(range(1, 513), 32),
        random.sample(range(1, 513), 32),
        random.sample(range(1, 513), 32),
    ]
    
    best_init = max(candidates, key=lambda s: _score_set(s))
    best_score = _score_set(best_init)
    
    current = list(best_init)
    current_score = best_score
    best_overall = list(current)
    best_overall_score = current_score
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() < deadline - 3.0:
        idx = random.randint(0, 31)
        old_val = current[idx]
        new_val = random.randint(1, 512)
        
        if new_val in current:
            continue
        
        current[idx] = new_val
        new_score = _score_set(current)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_overall_score:
                best_overall_score = new_score
                best_overall = list(current)
        else:
            current[idx] = old_val
        
        T *= cooling
        if T < min_T:
            T = 5.0
            # Occasionally restart from best to escape local optima
            if random.random() < 0.3:
                current = list(best_overall)
                current_score = best_overall_score
    
    return best_overall
