"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    # Set random seed for reproducibility in local tests, but use time or fixed for eval?
    # The task says "Maximize set size".
    # We will use a randomized local search / simulated annealing approach.
    
    def get_subset_sums(elements):
        sums = {0}
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            sums = sums | new_sums
        sums.discard(0)
        return sums

    def is_valid(elements):
        if not elements:
            return True
        n = len(elements)
        sums = get_subset_sums(elements)
        
        # Check uniqueness
        if len(sums) != (1 << n) - 1:
            return False
            
        # Check divisibility
        sorted_sums = sorted(sums)
        # Optimization: if si is large, it can't divide small sj
        for i in range(len(sorted_sums)):
            si = sorted_sums[i]
            # If si is 1, it divides everything. So 1 cannot be a sum.
            # Since elements are >= 1, min sum >= 1.
            # If 1 is in sums, and there are other sums, invalid.
            if si == 1 and len(sums) > 1:
                return False
                
            for j in range(i + 1, len(sorted_sums)):
                sj = sorted_sums[j]
                if sj % si == 0:
                    return False
        return True

    # We want to maximize size.
    # Try sizes from 15 down to 1.
    # For each size, try many random configurations.
    
    # To speed up, we can bias towards larger numbers.
    # Small numbers are more likely to divide others.
    
    best_set = []
    
    # Try to find a set of size N
    # We'll use a simple random restart strategy with a time limit.
    # Since we don't know the max N, we'll try 15, 14, ...
    # But checking N=15 is slow.
    # Let's estimate: 2^15 = 32768 sums. Pairs ~ 5e8. Too slow in Python.
    # So we should limit N to 12 or 13 for the search.
    # If we find a valid set of size 12, that's likely good.
    # The current score 2.33 suggests size ~7 or 8 if seed is 3.
    # Let's aim for 10-12.
    
    max_size_to_try = 12
    attempts_per_size = 5000
    
    import time
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    for n in range(max_size_to_try, 0, -1):
        for _ in range(attempts_per_size):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random set of size n
            # Bias towards larger numbers to avoid small divisors
            # Pick from 1..256
            # Let's pick 50% from 1..128 and 50% from 129..256?
            # Or just uniform?
            # Uniform might hit small numbers often.
            # Let's try uniform first, it's simpler.
            
            elements = random.sample(range(1, 257), n)
            
            if is_valid(elements):
                # Found a valid set of size n
                # Can we try to extend it?
                # For now, return it.
                # But we might find a larger one later?
                # We are iterating n downwards, so first found is max.
                return [int(x) for x in elements]
        
        # If time is up, break
        if time.time() - start_time > time_limit:
            break
            
    # If no set found, return a known small valid set
    # {2, 3} is valid. Size 2.
    # {100, 101} is valid. Size 2.
    # Let's try to return the best found or a fallback.
    # We didn't track best found in the loop above, we returned immediately.
    # Let's modify to track best.
    
    # Re-implement with tracking best
    best = []
    start_time = time.time()
    
    # Try sizes 12 down to 1
    for n in range(12, 0, -1):
        # If we already have a set of size n, we can stop?
        # No, we want the largest.
        # So we try n=12. If found, return.
        # If not, try n=11, etc.
        
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                return [int(x) for x in best] if best else [100, 101]
            
            elements = random.sample(range(1, 257), n)
            
            if is_valid(elements):
                return [int(x) for x in elements]
                
    # Fallback
    return [100, 101]

# EVOLVE-BLOCK-END
