import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    target = factorial(12)
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Exponents: (10, 5, 2, 1, 1)
    target_exp = (10, 5, 2, 1, 1)
    primes = [2, 3, 5, 7, 11]
    
    # Generate all divisors of target
    divisors = [1]
    for p, e in zip(primes, target_exp):
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
        
    # Filter divisors > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # We want to find a subset of valid_divisors that multiplies to target.
    # We want to minimize the maximum element in the subset.
    # We iterate M (the maximum allowed factor) from the smallest valid divisor upwards.
    
    # Memoization for DFS: (rem_exp_tuple, next_index) -> bool
    # But since we change the candidate list for each M, we can't share memo easily across M iterations 
    # unless we are careful. However, the search space is small enough that we might not need complex memoization 
    # or we can just use a set of visited states within one M iteration.
    
    # Actually, a better approach:
    # Just iterate M. For each M, get candidates = [d for d in valid_divisors if d <= M].
    # Check if any subset of candidates multiplies to target.
    
    # To check subset product efficiently:
    # Use DFS with backtracking.
    # State: (idx, rem_exp)
    # rem_exp is a tuple of remaining exponents.
    
    def can_form(rem_exp, idx, candidates):
        if rem_exp == (0, 0, 0, 0, 0):
            return True
        if idx >= len(candidates):
            return False
        
        # Pruning: if product of all remaining candidates < target? 
        # Hard to check with exponents directly without converting back to number, 
        # but we can just rely on the index bound and divisibility.
        
        for i in range(idx, len(candidates)):
            d = candidates[i]
            # Check if d divides the number represented by rem_exp
            # We need to subtract exponents of d from rem_exp
            # d = 2^a * 3^b * 5^c * 7^d * 11^e
            # We need to calculate exponents of d.
            # To avoid recalculating, we can precompute exponents for all valid_divisors.
            
            pass # Placeholder
            
    # Let's precompute exponents for all valid divisors
    div_exp = {}
    for d in valid_divisors:
        exps = [0]*5
        temp = d
        for i, p in enumerate(primes):
            while temp % p == 0:
                exps[i] += 1
                temp //= p
        div_exp[d] = tuple(exps)
        
    # Now iterate M
    # We can speed this up by noticing that if a solution exists for M, 
    # it might exist for M-1? No, we are increasing M.
    # We want the SMALLEST M.
    
    # We can just iterate M over the unique values in valid_divisors.
    # There are at most 792 divisors, so at most 792 iterations.
    # For each M, the DFS is very fast.
    
    for M in valid_divisors:
        candidates = [d for d in valid_divisors if d <= M]
        # Check if we can form target using distinct elements from candidates
        # DFS
        found = False
        # To avoid deep recursion overhead, we can use a simple recursive function
        # State: (rem_exp, next_idx)
        
        # We can use a set to memoize failed states (rem_exp, next_idx)
        memo = set()
        
        def dfs(rem_exp, next_idx):
            if rem_exp == (0, 0, 0, 0, 0):
                return True
            if next_idx >= len(candidates):
                return False
            if (rem_exp, next_idx) in memo:
                return False
            
            # Try skipping candidates[next_idx]
            # And trying candidates[next_idx]
            
            # Option 1: Skip
            if dfs(rem_exp, next_idx + 1):
                return True
            
            # Option 2: Take
            d = candidates[next_idx]
            d_exp = div_exp[d]
            
            # Check if we can subtract d_exp from rem_exp
            new_rem = [0]*5
            valid = True
            for i in range(5):
                if rem_exp[i] < d_exp[i]:
                    valid = False
                    break
                new_rem[i] = rem_exp[i] - d_exp[i]
            
            if valid:
                if dfs(tuple(new_rem), next_idx + 1):
                    return True
            
            memo.add((rem_exp, next_idx))
            return False
            
        if dfs(target_exp, 0):
            # We found a solution with max factor M.
            # We need to retrieve the actual factors.
            # We can modify dfs to return the list of factors.
            
            # Let's re-run dfs to get the path
            path = []
            def dfs_path(rem_exp, next_idx):
                if rem_exp == (0, 0, 0, 0, 0):
                    return True
                if next_idx >= len(candidates):
                    return False
                if (rem_exp, next_idx) in memo:
                    return False
                
                # Try taking
                d = candidates[next_idx]
                d_exp = div_exp[d]
                new_rem = [0]*5
                valid = True
                for i in range(5):
                    if rem_exp[i] < d_exp[i]:
                        valid = False
                        break
                    new_rem[i] = rem_exp[i] - d_exp[i]
                
                if valid:
                    path.append(d)
                    if dfs_path(tuple(new_rem), next_idx + 1):
                        return True
                    path.pop()
                
                # Try skipping
                if dfs_path(rem_exp, next_idx + 1):
                    return True
                
                memo.add((rem_exp, next_idx))
                return False
                
            dfs_path(target_exp, 0)
            return path
            
    # If no solution found (should not happen), return trivial
    return [target]
