"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Finds a large subset of [1, 100] with no subset summing to 200.
    Uses a greedy algorithm with bitset optimization for speed.
    """
    import time
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    def greedy_solve(random_seed=None):
        """
        Runs the greedy algorithm. If random_seed is provided, 
        it introduces slight randomization in the order of consideration 
        to explore different local optima.
        """
        import random
        if random_seed is not None:
            random.seed(random_seed)
        
        # We want to maximize cardinality. 
        # Standard greedy: iterate 100 down to 1.
        # If randomization is enabled, we might shuffle the lower numbers 
        # or skip some high numbers to see if a different combination works.
        # However, the descending order is the most critical part for this problem.
        # We will stick to descending order for the main pass to ensure stability,
        # but we can try a few variations.
        
        selected = []
        # Bitset of reachable sums. Bit i is 1 if sum i is reachable.
        # We only care about sums <= 200.
        # Python integers are arbitrary precision, so we can use a large int.
        # Mask to keep only bits 0..200.
        MASK = (1 << 201) - 1
        
        reachable = 1  # Bit 0 is set (sum 0)
        
        # Iterate from 100 down to 1
        # To allow for slight variations, we can group numbers or just run standard.
        # For now, standard descending.
        for x in range(100, 0, -1):
            # Check if adding x creates sum 200
            # new_reachable = reachable | (reachable << x)
            # If bit 200 is set in new_reachable, we cannot add x.
            
            # Optimization: Only check if (reachable >> (200-x)) & 1 is 1?
            # No, because we need to update reachable if we add it.
            # But we can check conflict first.
            # Conflict exists if there is a sum s in reachable such that s + x = 200.
            # i.e., s = 200 - x.
            # So if bit (200-x) is set in reachable, conflict.
            
            if x <= 200:
                if (reachable >> (200 - x)) & 1:
                    # Conflict, skip x
                    continue
            
            # No conflict, add x
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= MASK
            
            # Time check
            if time.time() > deadline:
                break
        
        return selected

    # Primary run: Standard Greedy
    best_set = greedy_solve()
    best_size = len(best_set)
    
    # If we have time, try a few randomized restarts to see if we can beat the greedy result.
    # The greedy result is a local optimum.
    # Randomization strategy: Shuffle the order of numbers 1..100? 
    # No, descending is key. 
    # Maybe skip the first few large numbers?
    # Or try a different order for small numbers?
    
    # Let's try a few variations where we randomly exclude one of the top 10 numbers 
    # and see if we can compensate with more smaller numbers.
    # Actually, the greedy is very strong. 
    # We will just ensure the bitset version is correct and fast.
    
    # Verify the result is valid (just in case)
    def is_valid(subset):
        # Check all subsets? Too slow for large sets.
        # Check using bitset again.
        reach = 1
        MASK = (1 << 201) - 1
        for x in subset:
            reach |= (reach << x)
            reach &= MASK
        return (reach >> 200) & 1 == 0

    if not is_valid(best_set):
        # Fallback to a known valid set: {1..19}
        best_set = list(range(1, 20))
        
    return best_set

# EVOLVE-BLOCK-END
