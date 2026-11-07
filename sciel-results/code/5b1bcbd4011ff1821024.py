import math
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = divisors[0]
    for d in divisors[1:]:
        L = lcm(L, d)
    
    best_range = float('inf')
    best_vals = None
    
    # Search over base points in one LCM period
    for b in range(1, L + 1):
        # For each divisor, get the two closest multiples to b
        candidates = []
        for d in divisors:
            k = b // d
            c1 = k * d
            c2 = (k + 1) * d
            # Ensure positive
            if c1 < 1:
                c1 = d
            candidates.append((c1, c2))
        
        current_vals = []
        
        def backtrack(idx, cur_min, cur_max):
            nonlocal best_range, best_vals
            
            if idx == n:
                r = cur_max - cur_min
                if r < best_range:
                    best_range = r
                    best_vals = current_vals[:]
                return

            c1, c2 = candidates[idx]
            
            # Try c1
            if c1 not in current_vals:
                new_min = min(cur_min, c1)
                new_max = max(cur_max, c1)
                if new_max - new_min < best_range:
                    current_vals.append(c1)
                    backtrack(idx + 1, new_min, new_max)
                    current_vals.pop()
            
            # Try c2
            if c2 != c1 and c2 not in current_vals:
                new_min = min(cur_min, c2)
                new_max = max(cur_max, c2)
                if new_max - new_min < best_range:
                    current_vals.append(c2)
                    backtrack(idx + 1, new_min, new_max)
                    current_vals.pop()

        backtrack(0, float('inf'), -float('inf'))
    
    # Return as list of ints
    return [int(v) for v in best_vals]
