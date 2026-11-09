import time
from math import factorial

N = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # We want to find the minimum M such that there exists a set of distinct integers
    # f_i in (12, M] with product N.
    
    # Lower bound for M:
    # The factors must be > 12.
    # Smallest distinct factors > 12 are 13, 14, ...
    # But they must divide N.
    
    def can_solve(M):
        # DFS to see if we can form N using distinct factors in (12, M]
        # We pick factors in increasing order to ensure distinctness easily.
        
        def dfs(rem, last_factor, current_factors):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return current_factors
            
            # If we have used 12 factors, we can't add more
            if len(current_factors) == 12:
                return None
            
            # Try next factors from last_factor + 1 to M
            for f in range(last_factor + 1, M + 1):
                if rem % f != 0:
                    continue
                
                # Pruning:
                # The remaining product is rem // f.
                # We have at most 12 - len(current_factors) - 1 slots left.
                # Let slots_left = 12 - len(current_factors) - 1
                # The max product we can form with distinct integers > f and <= M is:
                # M * (M-1) * ... * (M - slots_left + 1)
                # If rem // f > max_possible, then this f is too small (leaves too much remainder).
                # Since we iterate f upwards, rem//f decreases.
                # So if rem//f > max_possible, we should continue.
                
                slots_left = 12 - len(current_factors) - 1
                
                # Calculate max possible product for remaining slots
                # If slots_left is 0, max_prod = 1
                max_prod = 1
                for i in range(slots_left):
                    # The largest available numbers are M, M-1, ...
                    # Note: we are picking f, so next available must be > f.
                    # But for an upper bound on what *any* set of distinct numbers <= M can produce,
                    # using M, M-1... is a valid upper bound.
                    max_prod *= (M - i)
                    
                if rem // f > max_prod:
                    continue
                
                # Also, if rem // f is 1, we are done (handled by rem==1 check in next recursion? No, we check at start)
                # If rem // f < 1, impossible (but rem% f==0 and rem>=f so rem//f >= 1)
                
                # Recurse
                res = dfs(rem // f, f, current_factors + [f])
                if res is not None:
                    return res
                    
            return None
        
        return dfs(N, 12, [])

    # Search for minimum M
    # Start from 13
    # The optimal M is likely not huge.
    # If we find a solution for M, return it.
    # If we don't find one by some limit, return the best found or a fallback.
    
    best_solution = None
    # We know [N] is a valid solution (1 factor, N > 12). Max factor N.
    # But we want to minimize max factor.
    
    # Let's try M from 13 upwards.
    # If M is too small, no solution.
    # If M is large enough, there is a solution.
    
    # Upper bound for M: N itself.
    # But we can be smarter.
    # If we use 1 factor, M=N.
    # If we use 2 factors, M >= sqrt(N) ~ 21886.
    # If we use 3 factors, M >= cbrt(N) ~ 782.
    # If we use 4 factors, M >= 4th_root(N) ~ 197.
    # If we use 5 factors, M >= 5th_root(N) ~ 86.
    # If we use 6 factors, M >= 6th_root(N) ~ 33.
    # If we use 7 factors, M >= 7th_root(N) ~ 22.
    # If we use 8 factors, M >= 8th_root(N) ~ 15.
    
    # So M should be around 15-50.
    # Let's iterate M from 13 to 100.
    
    for M in range(13, 101):
        if time.time() > deadline:
            break
        sol = can_solve(M)
        if sol is not None:
            best_solution = sol
            break
            
    if best_solution is None:
        # Fallback: try larger M or return [N]
        # If we didn't find one in 100, try up to 1000?
        # Or just return [N] which is valid.
        # But we want to minimize.
        # Let's try a few more if time permits.
        for M in range(101, 1001, 10):
            if time.time() > deadline:
                break
            sol = can_solve(M)
            if sol is not None:
                best_solution = sol
                break
                
    if best_solution is None:
        return [N]
        
    return best_solution
