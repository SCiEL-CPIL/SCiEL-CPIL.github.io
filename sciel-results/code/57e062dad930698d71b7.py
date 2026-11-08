import numpy as np
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Initial set: try a few small sets to get a baseline
    # The seed gives [0,1,3]
    # Let's compute its score
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        # Compute sumset
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / denom
    
    # Start with the seed set
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Try a few hand-crafted sets
    test_sets = [
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 7, 15, 31, 63, 127, 255, 511],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45],  # triangular numbers
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],  # squares
    ]
    for ts in test_sets:
        s = compute_score(ts)
        if s > best_score:
            best_score = s
            best_set = list(ts)
    
    # Simulated annealing
    # Represent set as binary vector up to some max
    # We'll grow the set size and max value
    
    # Parameters
    max_elements = 50  # max number of elements in set
    max_value = 2000   # max value of elements
    
    # Current state: binary vector of length max_value+1
    # Start from best_set
    current_bits = np.zeros(max_value + 1, dtype=bool)
    for x in best_set:
        if x <= max_value:
            current_bits[x] = True
    
    current_score = best_score
    
    # SA parameters
    T0 = 0.1  # initial temperature
    T_min = 0.001
    alpha = 0.999  # cooling rate
    T = T0
    
    steps = 0
    max_steps = 100000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Generate a move: flip one bit (add or remove element)
        # Don't allow removing 0
        pos = random.randint(1, max_value)  # don't touch 0
        old_val = current_bits[pos]
        
        # Apply move
        current_bits[pos] = not old_val
        
        # Compute new score
        U = [i for i in range(max_value + 1) if current_bits[i]]
        if not U or 0 not in U or len(U) < 2:
            new_score = 0.0
        else:
            new_score = compute_score(U)
        
        # Accept or reject
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(U)
        else:
            # Revert
            current_bits[pos] = old_val
        
        # Cool down
        T = max(T_min, T * alpha)
    
    # Final polish: local search on best_set
    # Try adding/removing individual elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_set_set = set(best_set)
        max_u = max(best_set) if best_set else 0
        
        # Try adding elements
        for x in range(1, max_u * 2 + 10):
            if x not in best_set_set:
                candidate = best_set_set | {x}
                s = compute_score(list(candidate))
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = list(candidate)
                    improved = True
        
        # Try removing elements (not 0)
        for x in list(best_set):
            if x == 0:
                continue
            candidate = best_set_set - {x}
            if len(candidate) >= 2:
                s = compute_score(list(candidate))
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = list(candidate)
                    improved = True
    
    return sorted(best_set)
