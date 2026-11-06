from fractions import Fraction
import time

def construct():
    start = time.time()
    deadline = start + 45
    
    # Greedy initial solution
    def greedy():
        rem = Fraction(1, 1)
        result = []
        min_d = 5
        while rem > 0 and len(result) < 256:
            d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            result.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        return result if rem == 0 else None
    
    best = greedy()
    if best is None:
        raise RuntimeError("no solution")
    best_len = len(best)
    
    # Try shorter solutions
    for target in range(1, best_len):
        if time.time() > deadline:
            break
        sol = _search(target, deadline)
        if sol is not None:
            best = sol
            break
    
    return best

def _search(k, deadline):
    """Find k distinct denominators >= 5 with reciprocal sum 1."""
    
    def dfs(rem, min_d, current):
        if time.time() > deadline:
            return None
        if rem == 0:
            return list(current) if len(current) == k else None
        if len(current) == k:
            return None
        
        remaining = k - len(current)
        d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        for d in range(d_min, 10000001):
            if time.time() > deadline:
                return None
            frac = Fraction(1, d)
            if frac > rem:
                break
            new_rem = rem - frac
            if new_rem == 0:
                if remaining == 1:
                    return current + [d]
                continue
            if new_rem < 0:
                break
            
            # Pruning
            if remaining > 1:
                # Max sum of (remaining-1) distinct fractions >= d+1
                if new_rem > Fraction(remaining - 1, d + 1):
                    continue
                # Min sum: (remaining-1) * 1/10^7
                if new_rem < Fraction(remaining - 1, 10000000):
                    continue
            
            result = dfs(new_rem, d + 1, current + [d])
            if result is not None:
                return result
        return None
    
    return dfs(Fraction(1, 1), 5, [])
