import time
from math import gcd

def construct():
        start_time = time.time()
        time_limit = 50.0  # Safety margin before 60s limit
        
        best_solution = [8, 28, 1736]
        best_max_denom = 1736
        
        # Precompute harmonic numbers up to a reasonable limit, say 2000 or dynamic
        # But limit changes.
        
        def find_solution(limit):
            # Precompute H for 1..limit
            H = [0.0] * (limit + 1)
            for i in range(1, limit + 1):
                H[i] = H[i-1] + 1.0/i
            
            def dfs(n, d, min_d, path):
                # n/d is the remainder
                if n == 0:
                    return path
                if n < 0:
                    return None
                
                # Lower bound for next denominator
                lower = max(min_d, (d + n - 1) // n)
                
                if lower > limit:
                    return None
                
                # Upper bound check
                # We need sum_{j=d+1}^{limit} 1/j >= n/d
                # H[limit] - H[d] >= n/d
                # If not, break
                # Note: H[d] is sum_{1..d}
                # We need to check for the current d (which is the denominator of the remainder? No, d is the denominator of the fraction n/d).
                # Wait, the variable `d` in dfs is the denominator of the fraction n/d.
                # The next denominator we pick will be `x`.
                # The remaining sum after picking `x` is `r'`.
                # We need to form `r'` using denominators `> x`.
                # The max sum available is `H[limit] - H[x]`.
                # So we iterate `x` from `lower` to `limit`.
                # If `H[limit] - H[x] < r'`, we break?
                # No, `r'` depends on `x`.
                # `r' = n/d - 1/x = (nx - d) / (dx)`.
                # So we can't pre-check `H[limit] - H[x]` against a constant `r`.
                # We must calculate `r'` for each `x`.
                
                # However, we can prune `x` if `1/x > n/d` (handled by lower bound).
                # And if `H[limit] - H[x] < r'`, we skip this `x`?
                # Yes.
                # But `r'` decreases as `x` increases?
                # `r' = n/d - 1/x`. As `x` increases, `1/x` decreases, so `r'` increases.
                # So `r'` is increasing in `x`.
                # `H[limit] - H[x]` is decreasing in `x`.
                # So the condition `H[limit] - H[x] >= r'` becomes harder to satisfy as `x` increases.
                # If it fails for some `x`, it will fail for all larger `x`.
                # So we can break the loop.
                
                for x in range(lower, limit + 1):
                    # Check time
                    if time.time() - start_time > time_limit:
                        return None
                    
                    # Calculate remainder
                    # r' = n/d - 1/x
                    # num' = n*x - d
                    # den' = d*x
                    num_next = n * x - d
                    if num_next == 0:
                        return path + [x]
                    if num_next < 0:
                        # 1/x > n/d, but x >= ceil(d/n) ensures 1/x <= n/d.
                        # So num_next should be >= 0.
                        break 
                    
                    den_next = d * x
                    
                    # Reduce
                    g = gcd(num_next, den_next)
                    if g > 1:
                        num_next //= g
                        den_next //= g
                    
                    # Pruning: Can we form num_next/den_next with denoms in [x+1, limit]?
                    # Max sum = H[limit] - H[x]
                    # Required sum = num_next / den_next
                    if H[limit] - H[x] < num_next / den_next:
                        break
                    
                    res = dfs(num_next, den_next, x + 1, path + [x])
                    if res is not None:
                        return res
                
                return None
            
            return dfs(5, 31, 1, [])
        
        current_limit = best_max_denom - 1
        while current_limit >= 1:
            if time.time() - start_time > time_limit - 1.0:
                break
            sol = find_solution(current_limit)
            if sol:
                actual_max = max(sol)
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = sol
                current_limit = actual_max - 1
            else:
                break
                
        return best_solution
