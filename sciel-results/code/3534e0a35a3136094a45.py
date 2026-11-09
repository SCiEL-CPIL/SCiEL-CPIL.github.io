import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    def time_exceeded():
        return time.time() > deadline

    # Precompute prime factors for all numbers up to 20000 using a sieve
    max_sum = 20000
    prime_factors = [set() for _ in range(max_sum + 1)]
    for d in range(2, max_sum + 1):
        if not prime_factors[d]:  # d is prime
            for m in range(d, max_sum + 1, d):
                prime_factors[m].add(d)

    def count_primes(arr):
        """Count distinct primes dividing all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= prime_factors[arr[i] + arr[j]]
        return len(primes)

    # Strategy: Find a set of 12 distinct integers <= 10000 where all pairwise
    # sums are 7-smooth (only primes 2,3,5,7 divide them).
    # This guarantees at most 4 distinct primes.
    
    # Step 1: Build compatibility graph using bitsets for speed
    # Two numbers a, b are compatible if a+b is 7-smooth
    smooth_primes = {2, 3, 5, 7}
    
    # Precompute which sums are 7-smooth
    is_7smooth = [False] * (max_sum + 1)
    for s in range(2, max_sum + 1):
        if prime_factors[s].issubset(smooth_primes):
            is_7smooth[s] = True
    
    # Build adjacency as bitsets for fast intersection
    # We'll use a list of Python integers as bitsets (Python ints are arbitrary precision)
    # Or use numpy uint64 arrays. Let's use Python ints for simplicity and speed in intersection.
    
    # Actually, for 10000 nodes, a list of 10000 Python ints (each ~1250 bytes) is fine.
    # But intersection of 10000-bit integers is very fast in Python.
    
    # Let's try a different approach: since we want a clique of size 12,
    # and the graph is dense for small numbers, we can use a greedy + local search approach.
    
    # First, let's try to find a good base set using modular arithmetic.
    # If all numbers are congruent to r mod M, and 2r is divisible by all primes in M,
    # then all sums are divisible by M.
    
    best_arr = list(range(1, 13))
    best_count = count_primes(best_arr)
    
    # Try to find a set where all sums are 7-smooth
    # Key insight: if all numbers are even, all sums are even (prime 2)
    # if all numbers are multiples of 3, all sums are multiples of 3 (prime 3)
    # if all numbers are multiples of 6, all sums are multiples of 6 (primes 2,3)
    # We want to minimize the number of distinct primes.
    
    # Let's try: all numbers are multiples of 2. Then all sums are even.
    # But sums might have other prime factors.
    
    # Better: find numbers such that all pairwise sums are 7-smooth.
    # This is equivalent to finding a clique in the compatibility graph.
    
    # Let's use a backtracking search with forward checking on the compatibility graph.
    # But first, let's filter to candidate numbers that have many 7-smooth sums.
    
    # For each number x in 1..10000, count how many y in 1..10000 have x+y 7-smooth
    # Then sort by this count and try to build a clique from the top candidates.
    
    # This is expensive (10000 * 10000 = 100M operations). Let's optimize.
    
    # Alternative: use the fact that if x and y are both multiples of 2, x+y is even.
    # If x and y are both multiples of 3, x+y is multiple of 3.
    # So if we pick numbers that are all multiples of 6, all sums are multiples of 6.
    # But sums might still have other prime factors.
    
    # Let's try a greedy approach:
    # 1. Start with empty set
    # 2. Add numbers one by one, ensuring all pairwise sums are 7-smooth
    
    # Precompute for each number x, the set of numbers y such that x+y is 7-smooth
    # This is a 10000x10000 boolean matrix, which is 100M booleans = 100MB. Too much.
    
    # Instead, let's use a different strategy:
    # Try small moduli and find arithmetic progressions where all pairwise sums are 7-smooth.
    
    # For modulus m, if we pick numbers all congruent to r mod m,
    # then all sums are congruent to 2r mod m.
    # We need 2r mod m to be such that any number congruent to 2r mod m is 7-smooth.
    # This is hard to guarantee.
    
    # Let's try a simpler approach: local search.
    # Start with a random set of 12 numbers, then iteratively replace numbers to reduce prime count.
    
    import random
    random.seed(42)
    
    # Try multiple random starting points
    for trial in range(100):
        if time_exceeded():
            break
        
        # Generate a random candidate set
        # Bias towards small numbers since they're more likely to have smooth sums
        arr = set()
        while len(arr) < 12:
            x = random.randint(1, 1000)
            arr.add(x)
        arr = list(arr)
        
        count = count_primes(arr)
        if count < best_count:
            best_count = count
            best_arr = arr[:]
            if best_count <= 4:
                return best_arr
        
        # Local search: try replacing one element at a time
        improved = True
        while improved and not time_exceeded():
            improved = False
            for pos in range(12):
                old = arr[pos]
                # Try a few replacements
                for _ in range(50):
                    new = random.randint(1, 10000)
                    if new == old or new in arr:
                        continue
                    arr[pos] = new
                    new_count = count_primes(arr)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_arr = arr[:]
                            if best_count <= 4:
                                return best_arr
                    else:
                        arr[pos] = old
                if time_exceeded():
                    return best_arr
    
    # If we haven't found a good solution, try the modular arithmetic approach
    # Find numbers that are all multiples of 2, 3, 5, or 7
    
    # Try: all numbers are even
    arr = [2 * i for i in range(1, 13)]
    count = count_primes(arr)
    if count < best_count:
        best_count = count
        best_arr = arr[:]
    
    # Try: all numbers are multiples of 3
    arr = [3 * i for i in range(1, 13)]
    count = count_primes(arr)
    if count < best_count:
        best_count = count
        best_arr = arr[:]
    
    # Try: all numbers are multiples of 6
    arr = [6 * i for i in range(1, 13)]
    count = count_primes(arr)
    if count < best_count:
        best_count = count
        best_arr = arr[:]
    
    # Try: all numbers are multiples of 30
    arr = [30 * i for i in range(1, 13)]
    count = count_primes(arr)
    if count < best_count:
        best_count = count
        best_arr = arr[:]
    
    return best_arr
