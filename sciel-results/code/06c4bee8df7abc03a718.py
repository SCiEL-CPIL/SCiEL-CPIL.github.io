import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # We try to find a solution for n=34 first.
    # If that fails, we try smaller n.
    # However, local search on n=34 might be too hard.
    # Let's try a range of n.
    
    best_result = None
    best_n = 0
    
    # Try n from 34 down to 10
    # But we should allocate time wisely.
    # Maybe try 34 for 30s, 33 for 30s, etc?
    # Or just try 34 with a full budget?
    # If 34 is not solvable, 33 might be easier.
    
    # Let's try a few high n values.
    for target_n in [34, 33, 32, 31, 30]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Run local search for target_n
        result = local_search(target_n, time_limit - (time.time() - start_time))
        if result is not None:
            return result
            
    # If no high n found, fall back to a simple valid one if possible, or seed
    # The seed is n=9.
    # Let's try to find a valid one for n=20 quickly?
    # Or just return the seed to be safe?
    # The task says "Maximize n".
    # If we can't find >9, we should return the best we can.
    
    # Let's try a quick search for n=20
    if time.time() - start_time < time_limit - 5:
        result = local_search(20, 5.0)
        if result is not None:
            return result
            
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def local_search(n, time_budget):
    start_time = time.time()
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Initialize colors randomly
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Precompute 4-sets and 5-sets containing each triple?
    # Or just iterate over all 4-sets and 5-sets?
    # For incremental update, we need to know which 4-sets/5-sets contain a given triple.
    
    # Let's build the incidence structure.
    # For each triple t, list of 4-sets containing it.
    # For each triple t, list of 5-sets containing it.
    
    # This might use memory, but n=34 is small.
    # 4-sets: C(34,4) = 46376.
    # 5-sets: C(34,5) = 278256.
    # Each 4-set has 4 triples.
    # Each 5-set has 10 triples.
    
    # Instead of storing lists, we can just calculate the change on the fly?
    # No, we need to update the counts R(S) and B(S).
    
    # Let's store R_counts and B_counts.
    # R_counts[s] = number of red triples in 4-set s.
    # B_counts[s] = number of blue triples in 5-set s.
    
    # We need to map 4-sets and 5-sets to indices.
    sets_4 = list(combinations(range(n), 4))
    sets_5 = list(combinations(range(n), 5))
    
    set4_index = {s: i for i, s in enumerate(sets_4)}
    set5_index = {s: i for i, s in enumerate(sets_5)}
    
    num_sets_4 = len(sets_4)
    num_sets_5 = len(sets_5)
    
    R_counts = [0] * num_sets_4
    B_counts = [0] * num_sets_5
    
    # Initialize counts
    # For each triple, find all 4-sets and 5-sets containing it.
    # This is O(num_triples * (n-3 + C(n-3,2)))
    
    # To speed up, we can precompute for each triple the list of 4-sets and 5-sets.
    # triple_to_4sets[t] = [idx4_1, idx4_2, ...]
    # triple_to_5sets[t] = [idx5_1, idx5_2, ...]
    
    triple_to_4sets = [[] for _ in range(num_triples)]
    triple_to_5sets = [[] for _ in range(num_triples)]
    
    for s4 in sets_4:
        idx4 = set4_index[s4]
        # 4 triples in s4
        for t in combinations(s4, 3):
            idx_t = triple_index[t]
            triple_to_4sets[idx_t].append(idx4)
            
    for s5 in sets_5:
        idx5 = set5_index[s5]
        # 10 triples in s5
        for t in combinations(s5, 3):
            idx_t = triple_index[t]
            triple_to_5sets[idx_t].append(idx5)
            
    # Now compute initial R_counts and B_counts
    for i, t in enumerate(triples):
        c = colors[i]
        for idx4 in triple_to_4sets[i]:
            if c == 0: # Red
                R_counts[idx4] += 1
        for idx5 in triple_to_5sets[i]:
            if c == 1: # Blue
                B_counts[idx5] += 1
                
    # Count violations
    violations = sum(1 for r in R_counts if r == 4) + sum(1 for b in B_counts if b == 10)
    
    if violations == 0:
        return {'n': n, 'colors': colors}
        
    # Local Search
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        # Pick random triple
        i = random.randint(0, num_triples - 1)
        old_color = colors[i]
        new_color = 1 - old_color
        
        # Calculate delta violations
        delta_v = 0
        
        # Update R_counts
        for idx4 in triple_to_4sets[i]:
            if old_color == 0: # Was Red, now Blue
                R_counts[idx4] -= 1
                # If it was 4, violation removed
                if R_counts[idx4] == 3:
                    delta_v -= 1
                # If it was 3, now 2, no change
            else: # Was Blue, now Red
                R_counts[idx4] += 1
                # If it was 3, violation added
                if R_counts[idx4] == 4:
                    delta_v += 1
                
        # Update B_counts
        for idx5 in triple_to_5sets[i]:
            if old_color == 1: # Was Blue, now Red
                B_counts[idx5] -= 1
                # If it was 10, violation removed
                if B_counts[idx5] == 9:
                    delta_v -= 1
            else: # Was Red, now Blue
                B_counts[idx5] += 1
                # If it was 9, violation added
                if B_counts[idx5] == 10:
                    delta_v += 1
                    
        violations += delta_v
        
        # Acceptance
        if delta_v < 0:
            colors[i] = new_color
        else:
            if random.random() < math.exp(-delta_v / T):
                colors[i] = new_color
            else:
                # Reject: revert counts
                for idx4 in triple_to_4sets[i]:
                    if old_color == 0:
                        R_counts[idx4] += 1
                    else:
                        R_counts[idx4] -= 1
                for idx5 in triple_to_5sets[i]:
                    if old_color == 1:
                        B_counts[idx5] += 1
                    else:
                        B_counts[idx5] -= 1
                violations -= delta_v
                
        T *= cooling
        if T < T_min:
            T = T_min
            
        if violations == 0:
            return {'n': n, 'colors': colors}
            
    return None
