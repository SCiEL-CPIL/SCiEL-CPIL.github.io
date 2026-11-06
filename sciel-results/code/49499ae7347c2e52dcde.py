import math
import time
from functools import lru_cache

T12 = math.factorial(12)

def get_valid_divisors(n):
    """Generate all divisors of n that are > 12."""
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_power)
            p_power *= p
        divisors = new_divs
        
    # Filter and sort
    valid = sorted([d for d in divisors if d > 12])
    return valid

CANDIDATES = get_valid_divisors(T12)
N_CANDS = len(CANDIDATES)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # We iterate M from smallest to largest candidate.
    # For each M, we check if a subset of CANDIDATES[:idx_M+1] multiplies to T12.
    # If yes, we reconstruct the subset.
    
    for max_idx in range(N_CANDS):
        if time.time() > deadline:
            # Fallback: return a known valid solution
            # [T12] is always valid (1 factor, > 12)
            return [T12]
            
        M = CANDIDATES[max_idx]
        
        # Check if a decomposition exists using candidates up to index max_idx
        # We use a recursive search with memoization.
        # State: (remaining_product, current_index)
        # We clear the cache for each M to avoid stale data from different max_idx constraints?
        # Actually, the constraint is "index <= max_idx".
        # If we memoize based on (rem, idx), the result is valid for any max_idx >= idx.
        # So we can keep a global cache? 
        # If exists(rem, idx) returns True, it means there is a subset of C[idx:] that works.
        # This subset is valid for max_idx if the indices used are <= max_idx.
        # The recursion naturally enforces `idx` increases, so if we start with `max_idx`, 
        # and the recursion only goes forward, the indices used will be <= the starting max_idx 
        # ONLY IF we stop or fail when idx > max_idx.
        
        # Let's use a local cache for each M to be safe and simple, as N_CANDS is small.
        
        @lru_cache(maxsize=None)
        def can_form(rem, idx):
            if rem == 1:
                return True
            if idx > max_idx:
                return False
            if idx >= N_CANDS:
                return False
            
            c = CANDIDATES[idx]
            if c > rem:
                # Since CANDIDATES is sorted, all subsequent are >= c > rem.
                # We cannot pick any of them.
                return False
            
            # Option 1: Skip c
            if can_form(rem, idx + 1):
                return True
            
            # Option 2: Pick c
            if rem % c == 0:
                if can_form(rem // c, idx + 1):
                    return True
            
            return False

        if can_form(T12, 0):
            # Reconstruct the solution
            factors = []
            rem = T12
            idx = 0
            while rem > 1:
                if idx > max_idx:
                    break
                c = CANDIDATES[idx]
                if c > rem:
                    # Can't pick c, must skip
                    idx += 1
                    continue
                
                # Try to pick c
                if rem % c == 0:
                    # Check if picking c leads to a solution
                    # We need to verify if can_form(rem//c, idx+1) is True
                    if can_form(rem // c, idx + 1):
                        factors.append(c)
                        rem //= c
                        idx += 1
                        continue
                
                # If we didn't pick c (either didn't divide or didn't lead to solution), skip it
                idx += 1
                
            # Verify solution
            if math.prod(factors) == T12 and all(f > 12 for f in factors) and len(set(factors)) == len(factors):
                return factors
            else:
                # Should not happen if can_form was correct, but fallback
                return [T12]
                
    return [T12]
