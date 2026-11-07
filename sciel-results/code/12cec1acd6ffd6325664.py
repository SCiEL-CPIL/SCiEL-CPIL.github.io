import math

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
    
    # Sort indices by divisor size descending for better pruning
    # Larger divisors have larger gaps, so fixing them first constrains the range more
    sorted_indices = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    
    # Search over base points in one LCM period
    for b in range(1, L + 1):
        # For each divisor, get the two closest multiples to b
        candidates = [None] * n
        for i in range(n):
            d = divisors[i]
            k = b // d
            c1 = k * d
            c2 = (k + 1) * d
            # Ensure positive
            if c1 < 1:
                c1 = d
            candidates[i] = (c1, c2)
        
        # Backtracking search
        vals = [0] * n
        used = set()
        
        def backtrack(idx, cur_min, cur_max):
            nonlocal best_range, best_vals
            if idx == n:
                r = cur_max - cur_min
                if r < best_range:
                    best_range = r
                    best_vals = vals[:]
                return
            
            orig_idx = sorted_indices[idx]
            c1, c2 = candidates[orig_idx]
            
            # Try c1
            if c1 not in used:
                new_min = min(cur_min, c1)
                new_max = max(cur_max, c1)
                if new_max - new_min < best_range:
                    used.add(c1)
                    vals[orig_idx] = c1
                    backtrack(idx + 1, new_min, new_max)
                    vals[orig_idx] = 0
                    used.remove(c1)
            
            # Try c2
            if c2 not in used:
                new_min = min(cur_min, c2)
                new_max = max(cur_max, c2)
                if new_max - new_min < best_range:
                    used.add(c2)
                    vals[orig_idx] = c2
                    backtrack(idx + 1, new_min, new_max)
                    vals[orig_idx] = 0
                    used.remove(c2)
        
        # Initialize with infinite range
        backtrack(0, float('inf'), float('-inf'))
    
    # Return as list of ints
    if best_vals is None:
        # Fallback, should not happen
        return [d for d in divisors]
    return [int(v) for v in best_vals]
