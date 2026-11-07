import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 10_000_000
    
    # Initial fallback pair
    # sigma(31) = 32, sigma(21) = 32
    # |31/21 - 3/2| = |62/42 - 63/42| = 1/42
    best_a = 31
    best_b = 21
    best_err = Fraction(1, 42)
    
    def update(a, b):
        nonlocal best_a, best_b, best_err
        if a == b:
            return
        # a/b - 3/2 = (2a - 3b) / 2b
        # We want to minimize |2a - 3b| / (2b)
        # Since a, b > 0, b is positive.
        # We assume 2b is positive.
        err = abs(Fraction(2*a - 3*b, 2*b))
        if err < best_err:
            best_err = err
            best_a = a
            best_b = b
    
    # Try to find better pairs using a smaller N for speed, or optimized sieve
    # The previous program was too slow. Let's try N = 1_000_000 first.
    # If time permits, we can try larger N or a different strategy.
    
    try:
        N_search = 1_000_000
        # Use a list of ints for sigma. 
        # N=1e6, list of 1e6 ints is fine.
        sigma = [0] * (N_search + 1)
        
        # Sieve for sigma
        # Complexity O(N log N)
        # For N=1e6, this is roughly 1e6 * (ln(1e6) + gamma) ~ 1e6 * 14 ~ 1.4e7 operations.
        # In Python, this might take a few seconds.
        
        for i in range(1, N_search + 1):
            if time.time() - start_time > deadline - 5.0:
                break
            for j in range(i, N_search + 1, i):
                sigma[j] += i
        
        if time.time() - start_time > deadline - 2.0:
            return [best_a, best_b]
        
        # Group numbers by sigma value
        # We only care about groups where we can find a, b such that a/b is close to 3/2.
        # This implies a and b are of similar magnitude.
        # We can iterate through groups.
        
        # To avoid O(N^2) in large groups, we can use the property:
        # For a fixed b, we want a such that 2a ≈ 3b.
        # So a ≈ 1.5 b.
        # We can store for each sigma value, the set of numbers having that sigma.
        # Then for each number b in the set, check if there's an a in the set close to 1.5b.
        
        # Using a dictionary of sets
        groups = {}
        for n in range(1, N_search + 1):
            s = sigma[n]
            if s not in groups:
                groups[s] = []
            groups[s].append(n)
        
        # For each group, sort the numbers.
        # Then for each b, binary search for a closest to 1.5b.
        
        for s, nums in groups.items():
            if time.time() - start_time > deadline - 2.0:
                break
            if len(nums) < 2:
                continue
                
            nums.sort()
            
            # For each b in nums, find a in nums such that a/b is close to 3/2.
            # This is equivalent to 2a - 3b = 0.
            # Let's iterate b.
            for b in nums:
                target = 1.5 * b
                # Find index where target would be inserted
                # Since nums is sorted, we can use bisect
                import bisect
                idx = bisect.bisect_left(nums, target)
                
                # Check idx and idx-1
                if idx < len(nums):
                    a = nums[idx]
                    update(a, b)
                if idx > 0:
                    a = nums[idx-1]
                    update(a, b)
                
                # Also check a few neighbors? 
                # The error is |2a - 3b| / 2b.
                # As a moves away from 1.5b, the numerator grows linearly.
                # The denominator is fixed for a given b.
                # So the closest a is the best.
                
    except Exception as e:
        pass
    
    return [best_a, best_b]
