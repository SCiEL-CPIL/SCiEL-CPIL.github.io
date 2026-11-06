import time
from math import gcd

def construct():
    deadline = time.time() + 55
    
    # Initial best solution
    best = [8, 31, 248]
    best_max = 248
    
    # Precompute harmonic sums H[M] = sum(1/i for i in 1..M)
    # To check if sum_{i=d+1}^M 1/i >= R
    # sum_{i=d+1}^M 1/i = H[M] - H[d]
    # We need this >= R
    # We can use floating point for pruning, it's safe if we add a small epsilon
    # or just use it as a necessary condition.
    
    max_den = 10000000
    # But we only care up to 247 for improvement
    # Let's compute H up to 300
    H = [0.0] * 301
    for i in range(1, 301):
        H[i] = H[i-1] + 1.0/i
        
    def dfs(p, q, last, limit, path):
        # p/q is the remaining fraction
        # last is the last denominator used
        # limit is the max denominator allowed
        # path is the list of denominators used so far
        
        if time.time() > deadline:
            return None
            
        if p == 0:
            return list(path)
            
        if p < 0 or q <= 0:
            return None
            
        # If p/q is too small to be represented by any single term?
        # No, we can use multiple terms.
        
        # Lower bound for next denominator d
        # 1/d < p/q  => d > q/p
        d_lo = q // p + 1
        if d_lo <= last:
            d_lo = last + 1
            
        # Upper bound for next denominator d
        d_hi = limit
        
        # Pruning:
        # We need to form p/q using denominators > d and <= limit.
        # The max sum we can form is sum_{i=d+1}^{limit} 1/i.
        # If this sum < p/q, we cannot form p/q.
        # So we need H[limit] - H[d] >= p/q.
        # Note: H is precomputed up to 300. If limit > 300, we need to handle it.
        # But limit starts at 247, so we are safe.
        
        # Also, we need at least 1 term.
        # If d_hi < d_lo, return None
        
        for d in range(d_lo, d_hi + 1):
            if time.time() > deadline:
                return None
            
            # Check if we can form the remainder with denominators > d
            # Remainder R' = p/q - 1/d = (pd - q) / (qd)
            p_rem = p * d - q
            if p_rem <= 0:
                continue
                
            q_rem = q * d
            g = gcd(p_rem, q_rem)
            p_rem //= g
            q_rem //= g
            
            # Pruning: Can we form p_rem/q_rem with denominators in [d+1, limit]?
            # Max sum available: H[limit] - H[d]
            # We need H[limit] - H[d] >= p_rem / q_rem
            # Use float for speed
            if limit <= 300:
                max_sum = H[limit] - H[d]
                if max_sum < (p_rem / q_rem) - 1e-12:
                    continue
            else:
                # Approximation
                # ln(limit) - ln(d) >= p_rem/q_rem
                # This is loose, so we skip this check if limit > 300
                pass
            
            path.append(d)
            res = dfs(p_rem, q_rem, d, limit, path)
            if res is not None:
                return res
            path.pop()
            
        return None

    # Search for solution with max denominator < 248
    limit = 247
    while limit >= 1:
        if time.time() > deadline:
            break
            
        # Try to find a solution with max denominator <= limit
        res = dfs(5, 31, 0, limit, [])
        
        if res:
            # Found a solution
            max_den_found = max(res)
            if max_den_found < best_max:
                best_max = max_den_found
                best = res
            # Try to find a better one
            limit = max_den_found - 1
        else:
            # No solution found with this limit
            # We can break if we are sure no smaller limit will work?
            # Not necessarily, but if no solution with limit L, there is no solution with limit < L.
            # So we can break.
            break
            
    return best
