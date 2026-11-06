import math
import random
import time

def _is_dissociated(S):
    """
    Check if all subset sums of reciprocals of elements in S are distinct.
    Uses integer arithmetic with LCM to avoid floating point precision issues.
    """
    if not S:
        return True
    
    # Calculate LCM of all elements in S
    L = 1
    for x in S:
        L = L * x // math.gcd(L, x)
    
    # Convert to integer weights: w_i = L / x_i
    # The sum of reciprocals is distinct iff the sum of weights is distinct.
    weights = [L // x for x in S]
    
    # Generate all subset sums of weights
    # Start with sum 0
    sums = {0}
    for w in weights:
        # Create new sums by adding w to all existing sums
        # We can update the set in place if we iterate over a copy, 
        # or use a set update.
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        sums |= new_sums
        
    # If all 2^k sums are distinct, the set of sums should have size 2^k
    return len(sums) == (1 << len(S))

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy construction with random orders
    # We try to build a set by adding elements one by one if they don't cause collisions.
    # We repeat this many times with different random orders.
    
    pool = list(range(1, 23))
    
    # We will try to find a set of size at least 10 first, then try to maximize.
    # Since we want to maximize, we run many iterations.
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        random.shuffle(pool)
        
        current_set = []
        # Greedy add
        for x in pool:
            candidate = current_set + [x]
            if _is_dissociated(candidate):
                current_set = candidate
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we found a very large set, we might break or try to improve it specifically
        # For now, keep searching.
        
        # Optimization: If we found size 15+, it's likely near optimal, 
        # but let's keep trying to see if we can go higher.
        
    # Strategy 2: Local search on the best found set
    # Try to swap elements in best_set with elements from pool to increase size or maintain size with better properties?
    # Actually, if we just want max size, and we have a set of size K, 
    # we can try to add one more element by removing one and adding two? 
    # Or just try to find a larger set via random sampling of size K+1.
    
    # Let's do a second phase: Random sampling of larger sets
    # If best_size is, say, 12, try to find a set of size 13.
    
    target_size = best_size + 1
    if target_size <= 22:
        while time.time() - start_time < timeout:
            # Sample a random subset of size target_size
            subset = random.sample(pool, target_size)
            if _is_dissociated(subset):
                best_size = target_size
                best_set = subset
                # Try to go even higher
                target_size += 1
                if target_size > 22:
                    break
    
    # If no set was found (unlikely), return a known valid small set
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set

# EVOLVE-BLOCK-END
