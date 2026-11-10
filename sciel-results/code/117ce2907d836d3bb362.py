import random
import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_quadruples(candidate):
    count = 0
    for comb in combinations(range(12), 4):
        subset = [candidate[i] for i in comb]
        if is_sidon(subset):
            count += 1
    return count

def construct():
    # Start with a random set
    current = random.sample(range(129), 12)
    current_cost = count_sidon_quadruples(current)
    
    best = current[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.95
    
    import math
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s for safety
    
    while T > T_min and (time.time() - start_time) < timeout:
        # Generate neighbor
        candidate = current[:]
        
        # Strategy: Replace 1 element with a random new value
        i = random.randint(0, 11)
        while True:
            v = random.randint(0, 128)
            if v not in candidate:
                break
        candidate[i] = v
        
        candidate_cost = count_sidon_quadruples(candidate)
        
        if candidate_cost < current_cost:
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
                if best_cost == 0:
                    # Found optimal set (max Sidon size 3)
                    break
        else:
            delta = candidate_cost - current_cost
            if delta == 0:
                if random.random() < 0.5:
                    current = candidate
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current = candidate
                    current_cost = candidate_cost
        
        T *= cooling
        
    return best
