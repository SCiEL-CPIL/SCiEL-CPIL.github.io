"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    from random import Random

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(values, time_limit):
        n = len(values)
        best_S = set()
        best_size = 0
        
        # Heuristic 1: Odd numbers
        odds = {x for x in values if x % 2 != 0}
        # Verify odds is sum-free (it is)
        if len(odds) > best_size:
            best_S = odds
            best_size = len(odds)

        # Heuristic 2: Greedy with random restarts
        start_time = time.time()
        rng = Random(42) # Fixed seed for reproducibility in restarts, or use time? 
        # Using a fixed seed for the solver's internal randomness makes the result deterministic for a given time, 
        # but we want to explore. Let's use a counter or time-based seed for variety if we restart.
        # Actually, just use a local RNG.
        
        attempts = 0
        while time.time() - start_time < time_limit:
            # Create a random order
            vals_copy = values[:]
            rng.shuffle(vals_copy)
            
            S = set()
            forbidden = set()
            
            for x in vals_copy:
                # Check if x can be added
                # 1. 2x in S?
                if 2 * x in S:
                    continue
                # 2. x in forbidden?
                if x in forbidden:
                    continue
                
                # Add x
                S.add(x)
                
                # Update forbidden
                # New sums: x + s for s in S (including x itself? x+x=2x)
                # New diffs: x - s and s - x for s in S
                
                # We iterate over the OLD S plus x?
                # Actually, we just added x.
                # We need to add sums/diffs involving x and any element in S (including x).
                # But x is now in S.
                # So we iterate over S (which includes x).
                
                # Optimization: iterate over list(S) to avoid runtime error if we modified S?
                # We are not modifying S in the loop, just reading.
                # But we are adding to forbidden.
                
                # Note: x+x = 2x.
                # x+s = s+x.
                # x-s = -(s-x).
                
                # To avoid double work, we can just iterate.
                for s in S:
                    # Sum
                    forbidden.add(x + s)
                    # Diff
                    forbidden.add(x - s)
                    # s - x is same as -(x-s), so if we add x-s, do we need s-x?
                    # Yes, because if y = s-x is added later, s = x + y. Violation.
                    # And if y = x-s is added later, x = s + y. Violation.
                    # So both are needed.
                    forbidden.add(s - x)
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
            attempts += 1
        
        return list(best_S)

    results = []
    total_start = time.time()
    total_limit = 110.0
    
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time based on n^2
    weights = [60*60, 120*120, 240*240]
    total_weight = sum(weights)
    
    for i, (size, seed) in enumerate(instances):
        vals = instance(size, seed)
        # Calculate time limit for this instance
        # Leave 5 seconds buffer at the end for return
        remaining_global = total_limit - (time.time() - total_start)
        if remaining_global <= 0:
            # Out of time, return best found so far (which might be empty or partial)
            # But we must return 3 lists. 
            # If we run out of time, we should probably return the odd heuristic for remaining instances.
            pass
            
        # Proportional time allocation
        allocated_time = (weights[i] / total_weight) * total_limit
        # Adjust for already elapsed time?
        # Simpler: just run until global deadline.
        # But we want to prioritize.
        
        # Let's just run each instance with a time slice.
        # Slice = allocated_time - (time spent so far on this instance? No, this is the first run).
        
        # To be safe, let's just run the solver with a time limit that ensures we finish all 3.
        # We can't easily do dynamic allocation in a simple loop without knowing how long each takes.
        # But the cost is predictable.
        
        # Let's use a fixed fraction of the total time for each, scaled by weight.
        # But we must check global time.
        
        t_start = time.time()
        # Max time for this instance
        max_t = allocated_time * 0.95 # safety margin
        
        res = solve_instance(vals, max_t)
        results.append(res)
        
        # Check if we are running out of total time
        elapsed = time.time() - total_start
        if elapsed > total_limit - 5:
             # If we are close to deadline, and we have instances left,
             # we should stop solving and just return the odd heuristic for the rest.
             pass

    # Ensure we have 3 results
    while len(results) < 3:
         # Fallback
         idx = len(results)
         size, seed = instances[idx]
         vals = instance(size, seed)
         results.append([x for x in vals if x % 2 != 0])

    return results
# EVOLVE-BLOCK-END
