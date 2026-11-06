import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_score = 0
    best_set = None
    
    N = 512
    max_d = N // 2  # 256
    
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        # Initialize: mix of strategies
        if restart == 0:
            # Known-good: dense core + outlier
            S = set(list(range(1, 32)) + [47])
        elif restart < 15:
            # Dense core with random perturbation
            core = list(range(1, 32))
            outliers = random.sample(range(33, 513), 1)
            S = set(core + outliers)
        elif restart < 30:
            # Random with bias toward small numbers
            S = set(random.sample(range(1, 100), 20) + random.sample(range(100, 513), 12))
        else:
            S = set(random.sample(range(1, 513), 32))
        
        cur_score = _score_set_pairs(S)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Local search with simulated annealing
        T = 3.0
        T_min = 0.05
        cooling = 0.998
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Pick random element to remove
            x = random.choice(list(S))
            # Pick random element to add (bias toward small numbers)
            if random.random() < 0.3:
                y = random.randint(1, 100)
            else:
                y = random.randint(1, 512)
            
            if y in S:
                continue
            
            # Try swap
            S_new = S - {x} | {y}
            new_score = _score_set_pairs(S_new)
            
            delta = new_score - cur_score
            if delta > 0 or (delta >= -1 and random.random() < np.exp(delta / T)):
                # Accept
                S = S_new
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                no_improve += 1
            
            T = max(T_min, T * cooling)
        
        # Reset temperature for next restart
        T = 3.0
    
    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)


def _score_set_pairs(S):
    """
    Score a set by counting distinct d > 0 such that a, a+d, a+2d all in S.
    For a 32-element set, enumerate all pairs and check for the third element.
    O(n^2) where n=32, so ~500 operations per call.
    """
    elements = sorted(S)
    n = len(elements)
    if n < 3:
        return 0
    
    S_set = S
    covered = set()
    
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            a_plus_d = elements[j]
            d = a_plus_d - a
            a_plus_2d = a + 2 * d
            if a_plus_2d in S_set:
                covered.add(d)
    
    return len(covered)
