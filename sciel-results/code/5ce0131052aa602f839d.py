"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    from functools import lru_cache

    target = factorial(12)  # 479001600

    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
        
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # We want to find the smallest M such that there exists a set of distinct divisors
    # from (12, M] whose product is target.
    # We iterate M through valid_divs in increasing order.
    
    for M in valid_divs:
        # Candidates for this M
        candidates = [d for d in valid_divs if d <= M]
        if not candidates:
            continue
            
        # Sort candidates in descending order to help pruning
        candidates.sort(reverse=True)
        n = len(candidates)
        
        # Precompute suffix products for pruning
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        # Check if we can form target with distinct factors from candidates
        # We use DFS with memoization.
        # State: (idx, remaining)
        # idx: index in candidates list
        # remaining: product still needed
        
        @lru_cache(maxsize=None)
        def dfs(idx, remaining):
            if remaining == 1:
                return True
            if idx >= n:
                return False
            if remaining > suffix_prod[idx]:
                return False
            
            c = candidates[idx]
            
            # Option 1: Take c
            if remaining % c == 0:
                if dfs(idx + 1, remaining // c):
                    return True
            
            # Option 2: Skip c
            if dfs(idx + 1, remaining):
                return True
                
            return False

        if dfs(0, target):
            # Solution found with max factor M.
            # Now we need to extract the actual set of factors.
            # We re-run the DFS to collect the path.
            
            def find_set(idx, remaining, current_set):
                if remaining == 1:
                    return list(current_set)
                if idx >= n:
                    return None
                if remaining > suffix_prod[idx]:
                    return None
                
                c = candidates[idx]
                
                # Try taking c first (consistent with search)
                if remaining % c == 0:
                    current_set.append(c)
                    res = find_set(idx + 1, remaining // c, current_set)
                    if res is not None:
                        return res
                    current_set.pop()
                
                # Try skipping c
                res = find_set(idx + 1, remaining, current_set)
                if res is not None:
                    return res
                    
                return None

            result = find_set(0, target, [])
            if result:
                return result
                
    # Fallback (should not reach here)
    return [target]

# EVOLVE-BLOCK-END
