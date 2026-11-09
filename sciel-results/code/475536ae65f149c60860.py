import time
import random
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

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Start with a diverse initial set spread across [0, 128]
    # Use numbers that create many sum collisions
    current_set = list(range(12))
    current_val = largest_sidon_subset_size(current_set)
    best_set = current_set[:]
    best_val = current_val
    
    # Local search with single-element replacements
    # Expanded range: [0, 50] for more collision opportunities
    
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        improved = False
        
        # Try replacing each element with values in [0, 50]
        for i in range(12):
            if time.time() > deadline:
                break
                
            old_val = best_set[i]
            
            for new_val in range(0, 51):
                if new_val == old_val:
                    continue
                    
                # Check if new_val is already in the set
                if new_val in best_set:
                    continue
                    
                # Create new set
                new_set = best_set[:]
                new_set[i] = new_val
                
                val = largest_sidon_subset_size(new_set)
                
                if val < best_val:
                    best_val = val
                    best_set = new_set
                    improved = True
                    
        if not improved:
            # Try random perturbations: replace 2-3 elements at once
            num_replace = random.randint(2, 3)
            positions = random.sample(range(12), num_replace)
            new_set = best_set[:]
            for pos in positions:
                new_set[pos] = random.randint(0, 50)
            
            # Ensure distinctness
            if len(set(new_set)) == 12:
                val = largest_sidon_subset_size(new_set)
                if val < best_val:
                    best_val = val
                    best_set = new_set[:]
            
            # Occasionally do a full random restart from [0, 60]
            if random.random() < 0.05:
                cand = random.sample(range(0, 61), 12)
                val = largest_sidon_subset_size(cand)
                if val < best_val:
                    best_val = val
                    best_set = cand[:]
    
    return best_set
