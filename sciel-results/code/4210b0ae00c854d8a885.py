def construct():
    from fractions import Fraction
    import sys
    
    target = Fraction(5, 31)
    
    def greedy():
        r = target
        vals = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals
    
    fallback = greedy()
    max_fallback = max(fallback)
    limit = max_fallback - 1
    
    if limit < 1:
        return fallback

    # Precompute harmonic sums
    H = [Fraction(0, 1)] * (limit + 1)
    for i in range(1, limit + 1):
        H[i] = H[i-1] + Fraction(1, i)
        
    sys.setrecursionlimit(10000)
    
    def dfs(rem, min_d):
        if rem == 0:
            return []
        if min_d > limit:
            return None
        
        # Pruning: Max sum
        if min_d > 1:
            max_sum = H[limit] - H[min_d-1]
        else:
            max_sum = H[limit]
            
        if max_sum < rem:
            return None
            
        # Pruning: Min term
        if Fraction(1, min_d) > rem:
            return None
            
        start_d = min_d
        min_possible_d = (rem.denominator + rem.numerator - 1) // rem.numerator
        if min_possible_d > start_d:
            start_d = min_possible_d
            
        for d in range(start_d, limit + 1):
            term = Fraction(1, d)
            if term > rem:
                continue
            new_rem = rem - term
            if new_rem == 0:
                return [d]
            res = dfs(new_rem, d + 1)
            if res is not None:
                return [d] + res
        return None

    result = dfs(target, 1)
    
    if result is not None:
        if len(set(result)) == len(result) and sum(Fraction(1, d) for d in result) == target:
            return result
        else:
            return fallback
    else:
        return fallback
