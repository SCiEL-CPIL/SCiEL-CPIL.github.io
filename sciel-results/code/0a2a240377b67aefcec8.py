"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _initial_sums(a):
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            sums.add(a[i] + a[j])
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Start with the prescribed elements
    base = [1, 2, 5, 11]
    base_sums = _initial_sums(base)
    
    best_set = list(base)
    best_size = len(base)
    
    # Greedy construction to get a good initial solution
    a = list(base)
    sums = set(base_sums)
    
    # Try to extend greedily from 12 to 284
    for x in range(12, 285):
        new_sums = {x + y for y in a} | {2 * x}
        if len(new_sums) == len(a) + 1 and not (new_sums & sums):
            a.append(x)
            sums |= new_sums
    
    if len(a) > best_size:
        best_size = len(a)
        best_set = list(a)
    
    # Now try local search / improvement
    # We'll use a simple approach: try swapping elements or adding/removing
    # to see if we can do better
    
    # First, let's try a more systematic search
    # Use backtracking with pruning for the remaining positions
    
    # Actually, let me think about this differently.
    # The Sidon set condition: all pairwise sums a_i + a_j (i <= j) are distinct.
    # This is equivalent to saying all differences a_j - a_i (i < j) are distinct
    # (Golomb ruler property), but with the sums formulation it's slightly different.
    
    # Wait, actually for Sidon sets, the condition is that all sums a_i + a_j with i <= j
    # are distinct. This is NOT the same as Golomb rulers (which require distinct differences).
    
    # Let me verify: for a Sidon set, if a_i + a_j = a_k + a_l with (i,j) != (k,l),
    # then it's not Sidon. This is equivalent to saying the set is a B_2 sequence.
    
    # The relationship to Golomb rulers: A set is Sidon iff all pairwise sums are distinct.
    # A Golomb ruler requires all pairwise differences to be distinct.
    # These are related but not identical properties.
    
    # Actually, for sets of positive integers, the Sidon condition (distinct sums) 
    # is equivalent to the Golomb ruler condition (distinct differences) when 
    # we consider the set properly. Let me think...
    
    # If a_i + a_j = a_k + a_l, then a_i - a_k = a_l - a_j. So if sums are distinct,
    # differences might not be. But actually for the specific formulation here,
    # we just need distinct sums.
    
    # Let me proceed with the greedy approach and then try to improve.
    
    # Try randomized local search
    rng = random.Random(42)
    
    # Try to find a larger set using a different greedy order
    # or using backtracking
    
    # Let's try a branch and bound approach with time limit
    # But first, let's see what size we got from greedy
    
    # Try multiple greedy strategies
    strategies = []
    
    # Strategy 1: ascending order (already done)
    # Strategy 2: try to fill gaps
    # Strategy 3: randomized
    
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        rng2 = random.Random(seed)
        
        a2 = list(base)
        sums2 = set(base_sums)
        
        # Try adding elements in random order
        candidates = list(range(12, 285))
        rng2.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 10:
                break
            new_sums = {x + y for y in a2} | {2 * x}
            if len(new_sums) == len(a2) + 1 and not (new_sums & sums2):
                a2.append(x)
                sums2 |= new_sums
        
        if len(a2) > best_size:
            best_size = len(a2)
            best_set = list(a2)
    
    # Try local search: remove one element and try to add two
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        
        if len(best_set) <= best_size:
            break
            
        a3 = list(best_set)
        sums3 = _initial_sums(a3)
        
        # Try removing a random non-base element
        non_base_indices = [i for i, x in enumerate(a3) if x not in base]
        if non_base_indices:
            idx = rng.choice(non_base_indices)
            removed = a3.pop(idx)
            # Recompute sums
            sums3 = _initial_sums(a3)
            
            # Try to add two elements
            added = []
            for x in range(1, 285):
                if time.time() > deadline - 10:
                    break
                if x in a3 or x == removed:
                    continue
                new_sums = {x + y for y in a3} | {2 * x}
                if len(new_sums) == len(a3) + 1 and not (new_sums & sums3):
                    a3.append(x)
                    sums3 |= new_sums
                    added.append(x)
                    if len(added) == 2:
                        break
            
            if len(a3) > best_size:
                best_size = len(a3)
                best_set = list(a3)
    
    # Return the best set found
    return best_set
# EVOLVE-BLOCK-END
