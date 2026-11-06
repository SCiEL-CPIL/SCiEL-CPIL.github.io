import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    max_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        power = 1
        for _ in range(max_exponents[i] + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
        
    divisors.sort()
    divisors_set = set(divisors)
    
    # Precompute divisors of each divisor of N that are > 12
    # divisors_gt12[d] = list of divisors of d that are > 12 and divide N
    # This is used for fast lookup in backtracking
    divisors_gt12 = {}
    for d in divisors:
        divs = []
        for x in divisors:
            if x > 12 and d % x == 0:
                divs.append(x)
        divs.sort()
        divisors_gt12[d] = divs

    # We will do a DFS that builds a set of factors.
    # To minimize the largest factor, we can try to find a solution with a specific max_bound.
    # Or we can just find *any* solution and then try to improve it?
    # The task says "Minimize the LARGEST factor".
    # The current program does binary search.
    # Let's stick to binary search but optimize the `can_decompose` function significantly.
    
    # Key optimization for can_decompose:
    # 1. Use memoization? State is (remaining, min_factor, max_factor).
    #    remaining can be large, min_factor varies.
    #    However, 'remaining' is always a divisor of N. There are only 792 such values.
    #    min_factor is determined by the last chosen factor + 1.
    #    max_factor is fixed during a binary search step.
    #    So we can memoize (remaining, min_factor) for a specific max_factor?
    #    Or just (remaining, min_factor) globally?
    #    If (remaining, min_factor) is solvable with max_factor M, is it solvable with M' > M?
    #    Yes, because having a larger max_factor relaxes the constraint.
    #    So if we know it's solvable with M, it's solvable with any M' >= M.
    #    But we are checking for a specific M.
    #    Let's use a cache for the current binary search step.
    
    def solve_with_bound(max_factor):
        # Returns True if N can be decomposed into distinct factors > 12, all <= max_factor
        # We use DFS with pruning.
        
        # Memoization key: (remaining, min_factor)
        # Value: True/False
        # Note: If it returns False, it's False for this max_factor.
        # If it returns True, it's True for this max_factor.
        # Since we are doing binary search, we clear the cache for each max_factor check?
        # Or can we reuse?
        # If (rem, min_f) is False for M, it might be True for M+1.
        # So we must clear or scope the cache to the current M.
        
        memo = {}
        
        def dfs(remaining, min_factor, depth):
            if time.time() > deadline:
                return False
            
            if remaining == 1:
                return True
            
            if depth >= 12:
                return False
                
            if min_factor > max_factor:
                return False
            
            # Pruning: 
            # The product of the smallest possible distinct factors starting from min_factor
            # must be <= remaining?
            # Actually, remaining is the product of the factors we need to pick.
            # If we pick k factors, their product is remaining.
            # The smallest possible product of k distinct integers >= min_factor is min_factor * (min_factor+1) * ... * (min_factor+k-1).
            # We don't know k.
            # But we know that the factors must be <= max_factor.
            # So the maximum product we can form with factors <= max_factor is bounded?
            
            # Better pruning:
            # If remaining is a prime > max_factor, return False.
            # If remaining has no divisors in [min_factor, max_factor], return False.
            
            key = (remaining, min_factor)
            if key in memo:
                return memo[key]
            
            candidates = divisors_gt12[remaining]
            
            # Find candidates in [min_factor, max_factor]
            # Since candidates are sorted, we can binary search or just filter.
            # Filtering is fast enough for 792 items.
            valid_candidates = []
            for d in candidates:
                if d < min_factor:
                    continue
                if d > max_factor:
                    break
                valid_candidates.append(d)
                
            if not valid_candidates:
                memo[key] = False
                return False
            
            # Try candidates.
            # Order: 
            # If we pick a small d, remaining becomes large.
            # If we pick a large d, remaining becomes small.
            # Usually, picking larger factors reduces the depth faster.
            # But we need distinct factors > d.
            # Let's try descending order of d.
            # Why? If d is large, R' = remaining // d is small.
            # If R' is small, it might be 1 (success) or require few small factors.
            # If R' requires factors > d, and d is large, it's harder.
            # So maybe ascending is better? 
            # If d is small, R' is large. R' can be decomposed into larger factors easily.
            # Let's try ascending first. It allows the next factors to be larger, which is easier.
            
            for d in valid_candidates:
                R_prime = remaining // d
                if R_prime == 1:
                    memo[key] = True
                    return True
                
                # Next factors must be > d
                if dfs(R_prime, d + 1, depth + 1):
                    memo[key] = True
                    return True
                    
            memo[key] = False
            return False
            
        return dfs(N, 13, 0)

    # Binary search for the minimum M
    # Lower bound: 13
    # Upper bound: N
    low = 13
    high = N
    best_M = N
    found = False
    
    # We can tighten the lower bound.
    # The product of 13, 14, ..., 13+k must be <= N? No, the product IS N.
    # If we use k factors, the minimum possible largest factor is roughly N^(1/k).
    # But they must be distinct and > 12.
    # Let's just binary search.
    
    while low < high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        
        if solve_with_bound(mid):
            best_M = mid
            high = mid
            found = True
        else:
            low = mid + 1
            
    if not found:
        # Fallback: return a valid decomposition, even if not optimal
        # Trivial: [N] is valid if N > 12. N is huge.
        # But we want to minimize largest.
        # Let's try to find *any* solution quickly if binary search fails.
        # Or just return [N] to be safe?
        # The task says "Minimize the LARGEST factor".
        # If we can't find a better one, [N] has largest N.
        # But surely we can do better.
        
        # Let's try a greedy approach to find a valid solution quickly.
        # Greedy: pick the smallest possible factor > 12 that divides N.
        # This minimizes the factors, potentially leading to a larger last factor?
        # No, picking small factors leaves a large remainder, which might be a single large factor.
        # Example: 13 * (N/13). Largest is N/13.
        # If we pick 13, 14, ... we get smaller largest factor.
        
        # Let's run DFS without bound to find any solution, then maybe refine?
        # Or just use the best_M found. If none found, use a heuristic.
        
        # Heuristic: try to decompose into as many small factors as possible.
        # This tends to minimize the max factor.
        
        def find_any_solution(remaining, min_factor):
            if remaining == 1:
                return []
            if time.time() > deadline:
                return None
                
            candidates = divisors_gt12[remaining]
            valid_candidates = [d for d in candidates if d >= min_factor]
            
            if not valid_candidates:
                return None
                
            # Try smallest first
            for d in valid_candidates:
                R_prime = remaining // d
                if R_prime == 1:
                    return [d]
                sub = find_any_solution(R_prime, d + 1)
                if sub is not None:
                    return [d] + sub
            return None
            
        factors = find_any_solution(N, 13)
        if factors:
            return factors
        else:
            return [N]
            
    # Now we need to reconstruct the actual factors for best_M
    def find_factors(remaining, min_factor, max_factor):
        if remaining == 1:
            return []
            
        if time.time() > deadline:
            return None
            
        candidates = divisors_gt12[remaining]
        valid_candidates = [d for d in candidates if min_factor <= d <= max_factor]
        
        if not valid_candidates:
            return None
            
        for d in valid_candidates:
            R_prime = remaining // d
            if R_prime == 1:
                return [d]
            sub = find_factors(R_prime, d + 1, max_factor)
            if sub is not None:
                return [d] + sub
        return None
        
    factors = find_factors(N, 13, best_M)
    
    if factors is None:
        # This should not happen if solve_with_bound was consistent
        # Fallback
        def find_any_solution(remaining, min_factor):
            if remaining == 1:
                return []
            candidates = divisors_gt12[remaining]
            valid_candidates = [d for d in candidates if d >= min_factor]
            if not valid_candidates:
                return None
            for d in valid_candidates:
                R_prime = remaining // d
                if R_prime == 1:
                    return [d]
                sub = find_any_solution(R_prime, d + 1)
                if sub is not None:
                    return [d] + sub
            return None
        factors = find_any_solution(N, 13)
        if factors is None:
            return [N]
            
    return factors
