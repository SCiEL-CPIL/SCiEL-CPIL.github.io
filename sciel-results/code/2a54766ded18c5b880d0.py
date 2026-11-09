"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorization exponent vectors for 2..40
# Primes up to 40: 2,3,5,7,11,13,17,19,23,29,31,37
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES)}

def factorize(n):
    """Return tuple of exponents for each prime in PRIMES."""
    exponents = [0] * len(PRIMES)
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

# Precompute exponent vectors for all numbers 2..40
EXP_VECTORS = {}
for n in range(2, 41):
    EXP_VECTORS[n] = factorize(n)

# The key constraint: all subset sums of exponent vectors must be distinct.
# Equivalently, the set of exponent vectors must be "sum-distinct".
# A set S of vectors is sum-distinct if for any two disjoint subsets A, B of S,
# sum(A) != sum(B).

# For efficient checking during backtracking, we can use the fact that
# if we process vectors one by one, we need to check that the new vector
# doesn't create a collision with existing subset sums.

def has_distinct_subset_sums(vectors):
    """Check if all 2^n subset sums are distinct."""
    n = len(vectors)
    seen = set()
    seen.add(tuple(0 for _ in vectors[0]))
    for i in range(n):
        v = vectors[i]
        new_sums = set()
        for s in seen:
            new_s = tuple(s[j] + v[j] for j in range(len(v)))
            if new_s in seen or new_s in new_sums:
                return False
            new_sums.add(new_s)
        seen.update(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin before 90s kill
    
    # Candidate numbers 2..40, ordered by "difficulty" - 
    # numbers with more prime factors are more likely to cause collisions,
    # so try them later. Primes first (easiest), then prime powers, then composites.
    candidates = list(range(2, 41))
    
    # Sort by number of prime factors (with multiplicity), then by value
    def complexity(n):
        v = EXP_VECTORS[n]
        return sum(v)
    
    candidates.sort(key=lambda n: (complexity(n), n))
    
    # Backtracking search
    best = []
    best_len = 0
    
    # Use a recursive backtracking with the current set of vectors
    # and the set of all subset sums so far (for efficient collision check)
    
    def backtrack(idx, current_vectors, subset_sums, current_nums):
        nonlocal best, best_len
        
        if time.time() > deadline:
            return
        
        if len(current_nums) > best_len:
            best = list(current_nums)
            best_len = len(current_nums)
        
        if idx >= len(candidates):
            return
        
        # Try adding candidates[idx]
        n = candidates[idx]
        v = EXP_VECTORS[n]
        
        # Check if adding v creates a collision
        # New subset sums = old_sums union {s + v for s in old_sums}
        # Collision if any (s + v) is in old_sums or two different s1+v == s2+v
        # The second can't happen since addition is injective.
        # So we just need: for all s in subset_sums, (s + v) not in subset_sums
        
        new_sums = set()
        collision = False
        for s in subset_sums:
            new_s = tuple(s[j] + v[j] for j in range(len(v)))
            if new_s in subset_sums:
                collision = True
                break
            new_sums.add(new_s)
        
        if not collision:
            # Add it
            new_subset_sums = subset_sums | new_sums
            current_vectors.append(v)
            current_nums.append(n)
            
            backtrack(idx + 1, current_vectors, new_subset_sums, current_nums)
            
            current_vectors.pop()
            current_nums.pop()
        
        # Try skipping candidates[idx]
        backtrack(idx + 1, current_vectors, subset_sums, current_nums)
    
    # Start backtracking
    initial_sums = {tuple(0 for _ in PRIMES)}
    backtrack(0, [], initial_sums, [])
    
    # Verify the result
    if best:
        result = sorted(best)
        # Final verification with exact products
        def verify(nums):
            seen = set()
            n = len(nums)
            for mask in range(1 << n):
                prod = 1
                for i in range(n):
                    if mask & (1 << i):
                        prod *= nums[i]
                if prod in seen:
                    return False
                seen.add(prod)
            return True
        
        if verify(result):
            return result
    
    # Fallback: static construction from inspiration
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [4, 9, 16, 25]
    result = sorted(primes + composites)
    
    def verify2(nums):
        seen = set()
        n = len(nums)
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in seen:
                return False
            seen.add(prod)
        return True
    
    if verify2(result):
        return result
    
    return primes
# EVOLVE-BLOCK-END
