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

def get_max_sidon_size(candidate):
    n = len(candidate)
    # We want the max size.
    # Check from n down to 2
    for k in range(n, 1, -1):
        # If k is large, number of combinations is small.
        # If k is small (like 4), number is large.
        # But we stop as soon as we find one.
        # If the set is "bad" (small max sidon), we will check all k=12..k+1 and fail,
        # then check k.
        # If max sidon is 3, we check 12, 11, ..., 4 (all fail), then 3 (find one).
        # Number of checks:
        # k=12: 1
        # k=11: 12
        # k=10: 66
        # k=9: 220
        # k=8: 495
        # k=7: 792
        # k=6: 924
        # k=5: 792
        # k=4: 495
        # Total ~ 3800 checks.
        # Each check ~ 10-50 ops.
        # ~ 100k-200k ops per evaluation.
        # 5000 iterations -> 10^9 ops. Might be tight in Python.
        
        # Optimization: Prune early in is_sidon?
        # is_sidon already returns False early on collision.
        
        for comb in combinations(range(n), k):
            subset = [candidate[i] for i in comb]
            if is_sidon(subset):
                return k
    return 2

def construct():
    # Start with a random set
    current = random.sample(range(129), 12)
    current_score = get_max_sidon_size(current)
    
    best = current[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.99
    
    import math
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s for safety
    
    while T > T_min and (time.time() - start_time) < timeout:
        # Generate neighbor
        candidate = current[:]
        
        # Strategy: Swap one element with a random new value
        # or swap two elements
        if random.random() < 0.5:
            # Change one element
            i = random.randint(0, 11)
            # Pick a value not in candidate
            while True:
                v = random.randint(0, 128)
                if v not in candidate:
                    break
            candidate[i] = v
        else:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            # Note: swapping doesn't change the set of values, just positions.
            # Since the set is unordered, swapping positions doesn't change the set!
            # So this move is useless for the set content.
            # We must change the VALUES.
            # So we should replace one value with a new one, or replace two values with two new ones.
            
            # Let's stick to replacing one value with a new random one.
            # Or replace two values with two new random ones.
            
            # Let's refine the move:
            # 1. Replace 1 element with a random new one.
            # 2. Replace 2 elements with 2 random new ones.
            
            pass # The above swap logic was flawed for set optimization.

        # Corrected Move Logic:
        # We will just pick 1 or 2 indices and assign new random distinct values.
        
        # Re-do neighbor generation
        candidate = current[:]
        num_changes = 1 if random.random() < 0.7 else 2
        indices = random.sample(range(12), num_changes)
        for i in indices:
            while True:
                v = random.randint(0, 128)
                # Ensure v is not in candidate (except if it's the one we are replacing, 
                # but we are assigning, so we need distinctness)
                # Simple check:
                if v not in candidate:
                    break
            candidate[i] = v
        
        candidate_score = get_max_sidon_size(candidate)
        
        if candidate_score < current_score:
            current = candidate
            current_score = candidate_score
            if current_score < best_score:
                best = current[:]
                best_score = current_score
        else:
            # Delta is positive (worse) or zero
            delta = candidate_score - current_score
            if delta == 0:
                if random.random() < 0.5:
                    current = candidate
            else:
                # Accept if exp(-delta/T) > random
                # delta is integer (1, 2, ...)
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current = candidate
                    current_score = candidate_score
        
        T *= cooling
        
    return best
