"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a tuple of ints is a Sidon set (unique pair sums including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_subset(s):
    """Compute the largest Sidon subset of s using exhaustive search (fast for N=12)."""
    s = sorted(s)
    n = len(s)
    # We look for the largest k such that there exists a Sidon subset of size k
    for k in range(n, 0, -1):
        for comb in combinations(s, k):
            if is_sidon(comb):
                return k
    return 1

def construct():
    # Start with a dense arithmetic progression, which tends to have high sum collisions
    current = list(range(12))
    best_set = current[:]
    best_val = max_sidon_subset(best_set)
    
    start_time = time.time()
    timeout = 50
    current_val = best_val
    
    # Simulated Annealing / Random Local Search
    # We want to minimize current_val
    while time.time() - start_time < timeout:
        # Generate a neighbor
        candidate = current[:]
        i = random.randint(0, 11)
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in candidate and candidate[i] != new_val:
            continue
            
        candidate[i] = new_val
        
        # Validate distinctness (just in case of logic error above, though check is simple)
        if len(set(candidate)) != 12:
            continue
            
        val = max_sidon_subset(candidate)
        
        # Accept if better, or if equal with some probability, or worse with very low prob (SA)
        if val < current_val:
            current = candidate
            current_val = val
            if val < best_val:
                best_val = val
                best_set = candidate[:]
        elif val == current_val:
            if random.random() < 0.3:
                current = candidate
        else:
            # Occasionally accept worse to escape local minima
            if random.random() < 0.05:
                current = candidate
                current_val = val

    return best_set
# EVOLVE-BLOCK-END
