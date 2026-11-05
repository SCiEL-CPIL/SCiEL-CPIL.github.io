from fractions import Fraction
import time

def construct():
    start_time = time.time()
    
    # Try to find a solution with k=8
    # 1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g + 1/h = 1
    # a < b < c < d < e < f < g < h
    # a >= 5
    
    def dfs(rem, last, depth, k, path):
        if time.time() - start_time > 50:
            return None
        if depth == k:
            if rem == 0:
                return path[:]
            return None
        
        # Pruning: max possible sum of remaining terms
        # If rem is too large, return None
        # If rem is too small, return None
        
        # Next denominator d > last
        # 1/d <= rem => d >= 1/rem
        min_d = int(1 / rem)
        if min_d <= last:
            min_d = last + 1
        
        # Upper bound for d:
        # 1/d must be at least rem / (k - depth)
        # 1/d >= rem / (k - depth) => d <= (k - depth) / rem
        max_d = int((k - depth) / rem)
        
        if min_d > max_d:
            return None
            
        for d in range(min_d, max_d + 1):
            if rem < 0:
                break
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
            path.append(d)
            res = dfs(new_rem, d, depth + 1, k, path)
            if res:
                return res
            path.pop()
        return None

    for k in range(8, 12):
        res = dfs(Fraction(1, 1), 4, 0, k, [])
        if res:
            return res
            
    # Fallback to greedy
    target = Fraction(1, 1)
    denominators = []
    last = 4
    while target > 0:
        d = int(1 / target)
        if d <= last:
            d = last + 1
        while Fraction(1, d) > target:
            d += 1
        denominators.append(d)
        target -= Fraction(1, d)
        last = d
        if len(denominators) > 256:
            break
    return denominators
