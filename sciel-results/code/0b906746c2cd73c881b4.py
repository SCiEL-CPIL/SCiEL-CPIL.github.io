"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache
from itertools import product

def construct():
    target = math.factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (evaluator kills at 90s)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Primes: 2, 3, 5, 7, 11
    primes = [2, 3, 5, 7, 11]
    target_exponents = [10, 5, 2, 1, 1]
    
    # Precompute all divisors of target with their exponent vectors
    all_divisors = []
    all_divisor_set = set()
    
    # Generate all divisors by iterating over all combinations of exponents
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12 and d < target:  # exclude 1 and target itself for now
                            all_divisors.append(d)
                            all_divisor_set.add(d)
    
    # Also add target itself as a potential single factor
    all_divisors.append(target)
    all_divisor_set.add(target)
    
    # Sort divisors
    all_divisors.sort()
    
    # Map divisor to its exponent vector
    def get_exponents(n):
        exps = []
        for p in primes:
            cnt = 0
            while n % p == 0:
                cnt += 1
                n //= p
            exps.append(cnt)
        return tuple(exps)
    
    divisor_exponents = {}
    for d in all_divisors:
        divisor_exponents[d] = get_exponents(d)
    
    # For each divisor, store its exponent vector for quick subtraction
    # Precompute valid candidates: divisors > 12
    
    def can_decompose_csp(max_factor_limit):
        """Check if 12! can be decomposed into distinct factors > 12, all <= max_factor_limit.
           Returns the list of factors if found, else None."""
        
        # Get all candidate factors: divisors > 12 and <= max_factor_limit
        candidates = [d for d in all_divisors if 12 < d <= max_factor_limit]
        if not candidates:
            return None
        
        # Sort candidates for deterministic ordering
        candidates.sort()
        
        # Memoization: (remaining_exponents_tuple, min_candidate_index) -> bool
        memo = {}
        
        def dfs(remaining_exps, min_idx, chosen):
            if time.time() > deadline:
                return None
            
            # Check memo
            key = (remaining_exps, min_idx)
            if key in memo:
                return memo[key]
            
            # Base case: all exponents are zero
            if all(e == 0 for e in remaining_exps):
                memo[key] = list(chosen)
                return list(chosen)
            
            # Try each candidate from min_idx upward
            for idx in range(min_idx, len(candidates)):
                f = candidates[idx]
                f_exps = divisor_exponents[f]
                
                # Check if f can be subtracted from remaining
                can_subtract = True
                for i in range(len(primes)):
                    if remaining_exps[i] < f_exps[i]:
                        can_subtract = False
                        break
                
                if not can_subtract:
                    continue
                
                # Subtract and recurse
                new_exps = tuple(remaining_exps[i] - f_exps[i] for i in range(len(primes)))
                chosen.append(f)
                result = dfs(new_exps, idx + 1, chosen)
                chosen.pop()
                
                if result is not None:
                    memo[key] = result
                    return result
            
            memo[key] = None
            return None
        
        return dfs(tuple(target_exponents), 0, [])
    
    # Binary search on the maximum factor
    # Lower bound: 13 (smallest integer > 12)
    # Upper bound: target
    
    # But we need a better lower bound. The smallest possible max factor must be at least
    # the smallest divisor > 12 of 12!, which is 13? No, 13 doesn't divide 12!.
    # Let's find the smallest divisor of 12! that is > 12
    min_divisor = min(d for d in all_divisors if d > 12)
    
    # Binary search
    lo = min_divisor
    hi = target
    
    best_result = [target]
    
    # First, try to find a good upper bound with a few quick attempts
    # Try some reasonable values
    test_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    for tv in test_values:
        if tv >= hi:
            break
        if tv < lo:
            continue
        result = can_decompose_csp(tv)
        if result is not None:
            hi = tv
            best_result = result
            break
    
    # Now binary search between lo and hi
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = can_decompose_csp(mid)
        if result is not None:
            hi = mid
            best_result = result
        else:
            lo = mid + 1
    
    # Final check with the best found
    if time.time() < deadline - 2.0:
        final_result = can_decompose_csp(hi)
        if final_result is not None:
            best_result = final_result
    
    return best_result
# EVOLVE-BLOCK-END
