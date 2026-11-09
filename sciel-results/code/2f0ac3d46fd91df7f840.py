import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
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
            # Cannot add more terms if sum is already 0, unless depth is 0 (handled above)
            return None
        
        if rem < 0:
            return None

        # Calculate lower bound for the next denominator d
        # We need 1/d <= rem => d >= 1/rem
        # Also d >= min_d
        lower_bound = max(min_d, ceil_div(rem.denominator, rem.numerator))
        
        d = lower_bound
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > MAX_DENOM:
                break
            
            # Check if it's even possible to reach 'rem' with 'depth' terms starting from d
            # Max sum is sum(1/(d+i) for i in range(depth))
            # If this is less than rem, break.
            
            # Optimization: Calculate max sum efficiently for small depth
            # For depth <= 20, direct sum is fast.
            # For larger depth, use approximation or break early if depth/d < rem (loose bound)
            
            if depth <= 20:
                max_sum = sum(Fraction(1, d + i) for i in range(depth))
                if max_sum < rem:
                    break
            else:
                # Loose bound: sum(1/(d+i)) < depth/d
                if Fraction(depth, d) < rem:
                    break
                # If loose bound passes, we might still need to check more tightly, 
                # but for large depth, the solution space is likely different. 
                # However, minimal k is likely small.

            # Pick d
            new_rem = rem - Fraction(1, d)
            
            if new_rem < 0:
                d += 1
                continue
                
            if new_rem == 0:
                if depth == 1:
                    return path + [d]
                else:
                    # If depth > 1, we cannot have 0 remaining because we need distinct positive denominators
                    # and we haven't picked 'depth' terms yet.
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
    # k=1..7 are impossible as sum < 1 for max possible set.
    # Start from k=8.
    
    for k in range(8, 257):
        # Quick check if max possible sum < 1
        # For small k, calculate exactly.
        if k <= 20:
            max_sum = sum(Fraction(1, MIN_DENOM + i) for i in range(k))
            if max_sum < 1:
                continue
        else:
            # For large k, sum(1/(5+i)) grows like log(k).
            # log(256) + log(5) approx 5.5 + 1.6 = 7.1 > 1. So all k >= 8 are candidates.
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
