"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 1)
    return factors

def _parity_vector(n, prime_list):
    """Return tuple of parity (0/1) for each prime in prime_list."""
    factors = _factorize(n)
    vec = []
    for p in prime_list:
        vec.append(factors.get(p, 0) % 2)
    return tuple(vec)

def construct():
    deadline = time.time() + 85  # safety margin
    
    # Collect all primes that can appear in numbers 211..10210
    # For efficiency, we'll work incrementally
    # Start with a known good solution
    best_solution = [224, 240]
    best_raw = 30
    
    # Verify the known solution
    prod = 224 * 240 * 210
    r = math.isqrt(prod)
    if r * r != prod:
        # Fallback
        best_solution = [840]
        best_raw = 630
    
    # Try to find better solutions by increasing M from 211
    # For each M, check if there's a valid subset of {211,...,M}
    # Use BFS in parity state space
    
    # First, let's try small M values where we can do exhaustive-ish search
    for M in range(211, 240):
        if time.time() > deadline:
            break
        
        # Get all numbers in (210, M]
        nums = list(range(211, M + 1))
        
        # Collect all primes that appear in these numbers and in 210
        all_primes = set()
        factor_cache = {}
        for n in nums + [210]:
            f = _factorize(n)
            factor_cache[n] = f
            all_primes.update(f.keys())
        
        prime_list = sorted(all_primes)
        prime_to_idx = {p: i for i, p in enumerate(prime_list)}
        dim = len(prime_list)
        
        if dim > 60:
            # Too many dimensions for simple BFS, skip
            continue
        
        # Compute parity vectors
        vecs = {}
        for n in nums + [210]:
            f = factor_cache[n]
            vec = [0] * dim
            for p, exp in f.items():
                if exp % 2 == 1:
                    vec[prime_to_idx[p]] = 1
            vecs[n] = tuple(vec)
        
        target = vecs[210]
        
        # BFS: start from zero state, add numbers one by one
        # State: parity vector (as tuple)
        # We want to reach target state
        # Track: state -> (max_value_used, list_of_numbers)
        
        # Use a dict: state -> (max_val, frozenset_of_nums)
        # But this could explode. Let's limit to small subsets.
        
        # Try subsets of size 1, 2, 3, 4
        found = False
        
        # Size 1
        for n in nums:
            if vecs[n] == target:
                prod_check = n * 210
                r = math.isqrt(prod_check)
                if r * r == prod_check:
                    best_solution = [n]
                    best_raw = n - 210
                    found = True
                    break
        if found:
            continue
        
        # Size 2
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                # XOR of vecs
                v1 = vecs[nums[i]]
                v2 = vecs[nums[j]]
                xor_vec = tuple(a ^ b for a, b in zip(v1, v2))
                if xor_vec == target:
                    a, b = nums[i], nums[j]
                    prod_check = a * b * 210
                    r = math.isqrt(prod_check)
                    if r * r == prod_check:
                        best_solution = [a, b]
                        best_raw = max(a, b) - 210
                        found = True
                        break
            if found:
                break
        if found:
            continue
        
        # Size 3
        for i in range(len(nums)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    v1 = vecs[nums[i]]
                    v2 = vecs[nums[j]]
                    v3 = vecs[nums[k]]
                    xor_vec = tuple(a ^ b ^ c for a, b, c in zip(v1, v2, v3))
                    if xor_vec == target:
                        a, b, c = nums[i], nums[j], nums[k]
                        prod_check = a * b * c * 210
                        r = math.isqrt(prod_check)
                        if r * r == prod_check:
                            best_solution = [a, b, c]
                            best_raw = max(a, b, c) - 210
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if found:
            continue
        
        # Size 4
        for i in range(len(nums)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    for l in range(k + 1, len(nums)):
                        v1 = vecs[nums[i]]
                        v2 = vecs[nums[j]]
                        v3 = vecs[nums[k]]
                        v4 = vecs[nums[l]]
                        xor_vec = tuple(a ^ b ^ c ^ d for a, b, c, d in zip(v1, v2, v3, v4))
                        if xor_vec == target:
                            a, b, c, d = nums[i], nums[j], nums[k], nums[l]
                            prod_check = a * b * c * d * 210
                            r = math.isqrt(prod_check)
                            if r * r == prod_check:
                                best_solution = [a, b, c, d]
                                best_raw = max(a, b, c, d) - 210
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue
    
    # If we found a better solution, return it
    if best_raw < 30:
        return [int(x) for x in best_solution]
    
    # Otherwise return the known 2-element solution
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
