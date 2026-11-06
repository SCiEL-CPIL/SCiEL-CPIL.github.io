"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    target = math.factorial(12)  # 479001600
    deadline = time.time() + 50.0  # Safety margin
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    # Primes: [2, 3, 5, 7]
    primes = [2, 3, 5, 7]
    
    def prime_exponents(n):
        """Return tuple of exponents for primes in n."""
        exps = []
        for p in primes:
            e = 0
            while n % p == 0:
                e += 1
                n //= p
            exps.append(e)
        return tuple(exps)
    
    target_exps = prime_exponents(target)  # (10, 5, 2, 1)
    
    # Generate all divisors of 12! that are > 12
    # A divisor is determined by exponents (e2, e3, e5, e7) where 0<=e2<=10, 0<=e3<=5, 0<=e5<=2, 0<=e7<=1
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7)
                    if d > 12:
                        divisors.append((d, (e2, e3, e5, e7)))
    
    # Sort by divisor value
    divisors.sort(key=lambda x: x[0])
    div_values = [d[0] for d in divisors]
    div_exps = [d[1] for d in divisors]
    n_divs = len(divisors)
    
    # We want to find a subset of distinct divisors (by value) that multiply to target
    # and minimize the largest factor.
    
    # Strategy: iterate K over possible max values (div_values in increasing order)
    # For each K, check if there's a subset of divisors with value <= K that multiplies to target.
    
    # Use DFS with memoization on (remaining_exps, start_idx)
    # remaining_exps is a tuple of 4 ints
    
    def solve_for_K(K_idx):
        """Check if there's a valid subset using divisors[0..K_idx] that multiplies to target.
        Returns the list of divisors if found, else None."""
        
        # Memoize: (remaining_exps, start_idx) -> (found, path)
        # But path is expensive to memoize. Let's just memoize found/not-found.
        # Actually, we need the path. Let's use a different approach.
        
        # We'll use DFS that returns the path.
        # To avoid re-exploring, we memoize (remaining_exps, start_idx) -> True/False
        # and separately reconstruct the path.
        
        memo = {}
        
        def dfs(rem_exps, start_idx):
            """Returns True if we can form rem_exps using divisors[start_idx..K_idx]."""
            if rem_exps == (0, 0, 0, 0):
                return True
            
            key = (rem_exps, start_idx)
            if key in memo:
                return memo[key]
            
            # Pruning: if start_idx > K_idx, fail
            if start_idx > K_idx:
                memo[key] = False
                return False
            
            # Pruning: if remaining product < smallest available divisor, fail
            # Smallest available is div_values[start_idx]
            # But we need to compute the product from exponents... that's expensive.
            # Alternative: if rem_exps is "small" in some sense.
            # Let's just try each divisor.
            
            for i in range(start_idx, K_idx + 1):
                d_val = div_values[i]
                d_exp = div_exps[i]
                
                # Check if d_exp <= rem_exps (component-wise)
                valid = True
                for j in range(4):
                    if d_exp[j] > rem_exps[j]:
                        valid = False
                        break
                if not valid:
                    # Since divisors are sorted by value, larger divisors have larger or equal exponents?
                    # Not necessarily. E.g., 16 = 2^4 and 18 = 2*3^2.
                    # But if d_exp > rem_exps in some component, we can't use this divisor.
                    # We can't break early because a later divisor might be valid.
                    continue
                
                new_rem = tuple(rem_exps[j] - d_exp[j] for j in range(4))
                if dfs(new_rem, i + 1):
                    memo[key] = True
                    return True
            
            memo[key] = False
            return False
        
        if dfs(target_exps, 0):
            # Reconstruct the solution
            def reconstruct(rem_exps, start_idx):
                if rem_exps == (0, 0, 0, 0):
                    return []
                
                for i in range(start_idx, K_idx + 1):
                    d_val = div_values[i]
                    d_exp = div_exps[i]
                    
                    valid = True
                    for j in range(4):
                        if d_exp[j] > rem_exps[j]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_rem = tuple(rem_exps[j] - d_exp[j] for j in range(4))
                    if dfs(new_rem, i + 1):
                        return [d_val] + reconstruct(new_rem, i + 1)
                
                return None
            
            sol = reconstruct(target_exps, 0)
            if sol is not None:
                return sol
        
        return None
    
    # Iterate K in increasing order
    for K_idx in range(n_divs):
        if time.time() > deadline:
            break
        
        sol = solve_for_K(K_idx)
        if sol is not None:
            return sol
    
    # Fallback: return [target] if nothing found
    return [target]
# EVOLVE-BLOCK-END
