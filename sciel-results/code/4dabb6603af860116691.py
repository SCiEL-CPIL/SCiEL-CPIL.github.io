import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit (evaluator kills at 90s)
    
    MAX_DENOM = 10000000
    MIN_DENOM = 5

    def ceil_div(a, b):
        """Ceiling division for positive integers a, b."""
        return (a + b - 1) // b

    def dfs(rem, min_d, depth, path):
        """
        rem: remaining sum as Fraction
        min_d: minimum allowed denominator for the next term
        depth: number of terms left to choose
        path: list of chosen denominators so far
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem == 0:
                return path
            return None
        
        if rem == 0:
            return None
        
        if rem < 0:
            return None

        # Calculate lower bound for the next denominator d
        # We need 1/d <= rem => d >= 1/rem
        # Also d >= min_d
        lower_bound = max(min_d, ceil_div(rem.denominator, rem.numerator))
        
        # Upper bound logic:
        # We need to be able to form 'rem' with 'depth' terms, all >= d.
        # The maximum possible sum of 'depth' terms starting from d is S_max(d) = sum(1/(d+i) for i in 0..depth-1).
        # If S_max(d) < rem, then no solution exists for this d or any larger d (since S_max decreases as d increases).
        # So we can break the loop when this condition is met.
        
        # Also, we must have 1/d <= rem, which is satisfied by lower_bound.
        # And we must have 1/d + 1/(d+1) + ... >= rem.
        
        d = lower_bound
        
        # Optimization: Pre-calculate or approximate the max sum to break early?
        # For small depth (which is likely for minimal k), computing the sum is cheap.
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > MAX_DENOM:
                break
            
            # Check if it's even possible to reach 'rem' with 'depth' terms starting from d
            # Max sum is sum(1/(d+i) for i in range(depth))
            # If this is less than rem, break.
            
            # Approximation for speed: 
            # sum(1/(d+i)) <= depth/d
            # If depth/d < rem, break.
            # This is a loose bound but fast.
            if Fraction(depth, d) < rem:
                break
            
            # Tighter bound check for small depth to avoid unnecessary recursion
            # Only compute exact sum if depth is small enough or if the loose bound is close
            if depth <= 10:
                max_sum = sum(Fraction(1, d + i) for i in range(depth))
                if max_sum < rem:
                    break
            
            # Pick d
            new_rem = rem - Fraction(1, d)
            
            if new_rem < 0:
                d += 1
                continue
                
            if new_rem == 0:
                if depth == 1:
                    return path + [d]
                else:
                    # Cannot have 0 remaining with depth > 1 terms (denominators must be distinct and positive)
                    # Actually, if depth > 1 and new_rem == 0, we can't add more terms.
                    # So this path is invalid unless depth==1.
                    d += 1
                    continue

            new_min_d = d + 1
            new_depth = depth - 1
            new_path = path + [d]
            
            res = dfs(new_rem, new_min_d, new_depth, new_path)
            if res is not None:
                return res
            
            d += 1
            
        return None

    # Search for minimal k
    # We know k=1 is impossible (1/1, 1/2, etc. but min denom 5, so 1/5 != 1)
    # k=2: 1/a + 1/b = 1 => (a-1)(b-1)=1 => a=2, b=2 (not distinct, and <5)
    # k=3: 1/a+1/b+1/c=1. Min denom 5.
    #   1/5 + 1/6 + 1/30 = 6/30+5/30+1/30=12/30 != 1.
    #   1/5 + 1/5 ... distinct required.
    #   Let's just let the DFS handle it.
    
    # Pruning: Max sum with k terms starting from 5
    # If sum(1/(5+i) for i in range(k)) < 1, skip k.
    
    for k in range(2, 257):
        # Quick check if max possible sum < 1
        # Approximation: k/5 < 1 => k < 5. So k>=5 is needed?
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 = 0.2 + 0.166 + 0.142 + 0.125 + 0.111 = 0.744 < 1.
        # 1/5 + ... + 1/14 (10 terms) approx 0.2 + 0.16 + 0.14 + 0.125 + 0.11 + 0.1 + 0.09 + 0.083 + 0.077 + 0.071 = 1.05 > 1.
        # So k is likely around 10.
        
        # Exact check for small k
        if k <= 20:
            max_sum = sum(Fraction(1, 5 + i) for i in range(k))
            if max_sum < 1:
                continue
        else:
            # For large k, k/5 is a good upper bound on the sum? No, sum(1/(5+i)) < integral.
            # If k is large, the sum grows like log(k).
            # We can just proceed, DFS will prune quickly if impossible.
            pass

        result = dfs(Fraction(1), MIN_DENOM, k, [])
        if result is not None:
            return result
            
    # Fallback: Greedy algorithm
    # This guarantees a solution, though not minimal.
    def greedy(rem, min_d):
        path = []
        while rem > 0:
            if time.time() > deadline:
                return None
            # Find smallest d >= min_d such that 1/d <= rem
            d = max(min_d, ceil_div(rem.denominator, rem.numerator))
            if d > MAX_DENOM:
                return None
            path.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        return path

    return greedy(Fraction(1), MIN_DENOM)
