import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = factorial(12)
    max_factors = 12
    
    # Precompute divisors of N? Not strictly necessary for the logic, 
    # but helps to know if rem is a valid state.
    # rem must divide N.
    
    def solve_for_M(M):
        # Candidates from M down to 13
        candidates = list(range(M, 12, -1))
        
        # Quick check: if product of all candidates < N, impossible
        prod_all = 1
        for c in candidates:
            prod_all *= c
            if prod_all > N:
                break
        if prod_all < N:
            return None
            
        # Memoization table
        # Key: (idx, rem, count)
        # Value: True/False
        memo = {}
        choices = {}
        
        # Pre-check: if rem is not a divisor of N, it's a dead end.
        # We can filter states where rem does not divide N.
        
        def dfs(idx, rem, count):
            if time.time() > deadline:
                return False
                
            if rem == 1:
                return True
                
            if idx == len(candidates) or count >= max_factors:
                return False
                
            key = (idx, rem, count)
            if key in memo:
                return memo[key]
                
            # Pruning: if rem is not divisible by any remaining candidate, return False
            # This is expensive to check every time.
            # Instead, rely on the recursion.
            
            c = candidates[idx]
            
            # Try including c
            if rem % c == 0:
                if dfs(idx + 1, rem // c, count + 1):
                    memo[key] = True
                    choices[key] = c
                    return True
                    
            # Try excluding c
            if dfs(idx + 1, rem, count):
                memo[key] = True
                choices[key] = None
                return True
                
            memo[key] = False
            return False

        if dfs(0, N, 0):
            # Reconstruct
            factors = []
            idx = 0
            rem = N
            count = 0
            while rem != 1:
                key = (idx, rem, count)
                c = choices[key]
                if c is not None:
                    factors.append(c)
                    rem //= c
                    count += 1
                idx += 1
            return factors
        return None

    # Iterate M
    # Lower bound estimate:
    # (12!)^(1/12) approx 38.
    # Let's start from 13 to be safe, but it will be fast.
    for M in range(13, 200):
        res = solve_for_M(M)
        if res is not None:
            # Verify
            if len(res) <= 12 and all(f > 12 for f in res) and len(set(res)) == len(res):
                prod = 1
                for f in res:
                    prod *= f
                if prod == N:
                    return res
    return [N]
