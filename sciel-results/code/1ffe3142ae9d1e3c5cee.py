import time
import random
from math import prod

def is_valid(seq):
    """Check if all non-empty consecutive block products are distinct."""
    n = len(seq)
    if n == 0:
        return True
    products = set()
    # Generate all subarray products
    # O(N^3) is too slow for large N, but N is small (max 127).
    # Actually N is around 30-40. N^3 is 64k. Very fast.
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in products:
                return False
            products.add(p)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # 1. Generate primes and composites in [2, 128]
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(128)
    primes = [i for i in range(2, 129) if is_prime[i]]
    composites = [i for i in range(2, 129) if not is_prime[i]]
    
    # 2. Initialize best solution with primes
    # Primes always form a valid sequence
    best_seq = primes[:]
    best_len = len(best_seq)
    
    # 3. Randomized Greedy Search
    # We try to add composites to the prime base.
    # We perform multiple trials with different random orders of composites.
    
    # To speed up validity check, we can cache or optimize, but N is small.
    # is_valid is O(N^3). For N=40, 40^3 = 64,000 ops.
    # We might do thousands of checks. 64k * 1000 = 64M ops.
    # Python might be slow for 64M ops in 55s?
    # 10^7 ops/sec is typical for simple Python loops.
    # 64M might take ~6 seconds. So we can do ~10 trials of full greedy?
    # Or we can optimize is_valid.
    
    # Optimization for is_valid:
    # We can break early.
    
    def check_addition(current_seq, new_val):
        """
        Check if adding new_val to current_seq (which is sorted) results in a valid sequence.
        current_seq is sorted. new_val is inserted at the correct position.
        """
        # Find insertion index
        # Since current_seq is sorted, we can use bisect
        import bisect
        idx = bisect.bisect_left(current_seq, new_val)
        if idx < len(current_seq) and current_seq[idx] == new_val:
            return None # Duplicate value, not strictly increasing
            
        new_seq = current_seq[:idx] + [new_val] + current_seq[idx:]
        
        # Check validity of new_seq
        n = len(new_seq)
        products = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= new_seq[j]
                if p in products:
                    return None
                products.add(p)
        return new_seq

    # We will run multiple random trials
    # Each trial:
    # 1. Start with primes
    # 2. Shuffle composites
    # 3. Try to add each composite in order. If valid, add it.
    
    trials = 0
    while time.time() < deadline:
        trials += 1
        current_seq = primes[:]
        shuffled_composites = composites[:]
        random.shuffle(shuffled_composites)
        
        for c in shuffled_composites:
            # Try to add c
            new_seq = check_addition(current_seq, c)
            if new_seq is not None:
                current_seq = new_seq
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        # Optional: If we reach a very high length, we can stop early or tighten search
        # Max possible is 127.
        
    return best_seq
