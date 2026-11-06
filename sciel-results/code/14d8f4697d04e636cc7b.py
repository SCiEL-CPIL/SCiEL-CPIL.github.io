import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Known minimal solution for denominators >= 5
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This is 46 terms, which is valid but not minimal.
    
    # Let's try to find a smaller solution using DFS with tight pruning
    # The minimum number of terms for denominators >= 5 is at least 8
    # (since 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 0.852... < 1)
    # Actually, let's compute: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12 
    # = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 + 0.1 + 0.0909 + 0.0833 = 1.0199 > 1
    # So 8 terms might be possible.
    
    # Try DFS for k terms, starting from k=8
    for k in range(8, 20):
        if time.time() > deadline:
            break
        result = _dfs(k, 5, Fraction(1, 1), [])
        if result:
            return result
            
    # Fallback: use greedy algorithm to find any valid solution
    result = []
    remaining = Fraction(1, 1)
    min_d = 5
    
    for _ in range(256):
        if remaining == 0:
            break
        if len(result) >= 256:
            break
        
        # Find smallest d >= min_d such that 1/d <= remaining
        num, den = remaining.numerator, remaining.denominator
        d = max(min_d, (den + num - 1) // num)
        
        if d > 10**7:
            break
        
        result.append(d)
        remaining -= Fraction(1, d)
        min_d = d + 1
    
    if remaining == 0 and len(result) >= 1 and all(5 <= d <= 10**7 for d in result) and len(set(result)) == len(result):
        return result
    
    raise RuntimeError("no representation found")

def _dfs(k, start_denom, remaining, current):
    if k == 0:
        if remaining == 0:
            return current
        return None
    
    if remaining <= 0:
        return None
        
    # Pruning: max possible sum with k terms starting from start_denom
    max_possible = sum(Fraction(1, start_denom + i) for i in range(k))
    if remaining > max_possible:
        return None
        
    # Determine range for first denominator d
    # 1/d <= remaining => d >= ceil(1/remaining)
    min_d = max(start_denom, int(1 / remaining) + (1 if (1 % remaining) != 0 else 0))
    
    # Upper bound for d: we need remaining - 1/d to be representable by k-1 terms > d
    # Max sum of k-1 terms starting from d+1: sum(1/(d+1+i) for i in range(k-1))
    # We need remaining - 1/d <= max_next_sum
    # This gives a constraint on d
    
    # Iterate d from min_d upwards
    for d in range(min_d, 10000):  # Cap search for speed
        if d > 10**7:
            break
        if Fraction(1, d) > remaining:
            continue
        next_remaining = remaining - Fraction(1, d)
        if next_remaining < 0:
            continue
        
        # Check if next_remaining can be formed by k-1 terms > d
        if k - 1 == 0:
            if next_remaining == 0:
                return current + [d]
            continue
        
        # Max sum of k-1 terms starting from d+1
        max_next = sum(Fraction(1, d + 1 + i) for i in range(k - 1))
        if next_remaining > max_next:
            continue
            
        res = _dfs(k - 1, d + 1, next_remaining, current + [d])
        if res:
            return res
            
    return None
