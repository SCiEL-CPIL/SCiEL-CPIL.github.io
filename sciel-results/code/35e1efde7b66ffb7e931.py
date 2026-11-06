"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85
    
    target_num = 5
    target_den = 31
    
    def search(max_den):
        # Returns a list of denominators or None
        # Uses recursion with integer arithmetic
        
        def dfs(num, den, min_d, depth):
            if num == 0:
                return []
            
            if depth == 0:
                return None
            
            # Check if 1-step solution exists
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= max_den:
                    return [d]
            
            # d_min: 1/d < num/den => d > den/num
            d_min = max(min_d, den // num + 1)
            d_max = max_den
            
            if d_min > d_max:
                return None
            
            for d in range(d_min, d_max + 1):
                # Pruning: if d is too large, the remainder is too big to be covered by remaining terms
                # Condition: num/den - 1/d > (depth-1)/(d+1)
                # num/den > 1/d + (depth-1)/(d+1)
                # Multiply by d*(d+1)*den:
                # num * d * (d+1) > den * (d+1) + den * (depth-1) * d
                
                lhs = num * d * (d + 1)
                rhs = den * (d + 1) + den * (depth - 1) * d
                
                if lhs > rhs:
                    break # d is too large
                
                # Calculate remainder
                new_num = num * d - den
                new_den = den * d
                
                if new_num <= 0:
                    continue # Should not happen due to d_min, but safe
                
                # Simplify fraction
                g = math.gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                
                res = dfs(new_num, new_den, d + 1, depth - 1)
                if res is not None:
                    return [d] + res
            
            return None
        
        # Try depths 2 to 15
        # 2 terms is impossible for 5/31, but check just in case
        for depth in range(2, 16):
            if time.time() > deadline:
                return None
            res = dfs(target_num, target_den, 2, depth)
            if res is not None:
                return res
        
        return None

    # Iterate M from lower bound
    # 5/31 ~ 0.161. 1/7 ~ 0.142.
    # Min denominator must be at least 7.
    # We iterate M (max denominator) upwards.
    
    # To speed up, we can skip ranges.
    # But the pruning is strong, so linear scan is fine.
    
    for M in range(7, 10001):
        if time.time() > deadline:
            break
        
        sol = search(M)
        if sol is not None:
            # Verify
            s = sum(1/x for x in sol)
            if abs(s - 5/31) < 1e-9:
                return sol
        
    # Fallback: Greedy
    r = 5/31
    values = []
    while r > 0:
        d = int(1/r) + 1
        values.append(d)
        r -= 1/d
        if len(values) > 100:
            break
    
    return values
# EVOLVE-BLOCK-END
