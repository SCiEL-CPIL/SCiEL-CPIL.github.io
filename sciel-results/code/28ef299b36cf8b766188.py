"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # Map prime to index for packing
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Base for packing exponents
    # Max exponent of 2 in 40 is 5 (32)
    # Max exponent of 3 in 40 is 3 (27)
    # Max exponent of 5 in 40 is 2 (25)
    # Base 10 is safe and simple
    BASE = 10
    
    def get_packed_val(n):
        v = 0
        temp = n
        for p in primes:
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            if exp > 0:
                v += exp * (BASE ** prime_to_idx[p])
        return v

    # Precompute packed values for 2..40
    packed_vals = {}
    for n in range(2, 41):
        packed_vals[n] = get_packed_val(n)

    # Candidate numbers
    candidates = list(range(2, 41))

    best_set = []
    best_size = 0

    # Define a greedy function that takes an ordering
    def greedy(order):
        current_sums = {0}
        selected = []
        for n in order:
            if time.time() > deadline:
                return selected
            v = packed_vals[n]
            # New sums are current_sums union (current_sums + v)
            # Since we use packed integers, addition is valid
            new_sums = set()
            for s in current_sums:
                new_sums.add(s)
                new_sums.add(s + v)
            
            # If no collision, size doubles
            if len(new_sums) == 2 * len(current_sums):
                current_sums = new_sums
                selected.append(n)
        return selected

    # Try different orderings
    orderings = []
    
    # 1. Natural order
    orderings.append(candidates)
    
    # 2. Primes first
    primes_in_range = [n for n in candidates if n in prime_to_idx]
    composites_in_range = [n for n in candidates if n not in prime_to_idx]
    orderings.append(primes_in_range + composites_in_range)
    
    # 3. Composites first
    orderings.append(composites_in_range + primes_in_range)
    
    # 4. Random shuffles
    for _ in range(50):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
        
    # 5. Sorted by packed value (magnitude)
    orderings.append(sorted(candidates, key=lambda x: packed_vals[x]))
    
    # 6. Sorted by number of prime factors (complexity)
    def count_factors(n):
        c = 0
        temp = n
        for p in primes:
            while temp % p == 0:
                c += 1
                temp //= p
        return c
    orderings.append(sorted(candidates, key=count_factors))
    
    # Run greedy for each ordering
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    # If we have time, try local search / refinement on the best set
    # This is complex, so we might skip if time is tight.
    # But 50 random restarts should be fast enough.
    
    return best_set
# EVOLVE-BLOCK-END
