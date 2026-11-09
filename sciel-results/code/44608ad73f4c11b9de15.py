import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    MAX_DENOM = 10000000
    MIN_DENOM = 5
    
    def ceil_div(a, b):
        """Ceiling division for positive integers a, b."""
        return (a + b - 1) // b
    
    # Memoization for DFS states: (rem_num, rem_den, min_d, depth) -> True if failed
    memo = set()
    
    def dfs(rem_num, rem_den, min_d, depth, path):
        """
        rem_num/rem_den: remaining sum as reduced fraction
        min_d: minimum allowed denominator for the next term
        depth: number of terms left to choose
        path: list of chosen denominators so far
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem_num == 0:
                return path
            return None
        
        if rem_num == 0:
            return None
        
        if rem_num < 0:
            return None
        
        # Memoization check
        state = (rem_num, rem_den, min_d, depth)
        if state in memo:
            return None
        
        # Base case optimization: if only 1 term left, check if rem is a unit fraction
        if depth == 1:
            if rem_num == 1:
                d = rem_den
                if d >= min_d and d <= MAX_DENOM:
                    return path + [d]
            memo.add(state)
            return None
        
        # Calculate lower bound for the next denominator d
        # We need 1/d <= rem_num/rem_den => d >= rem_den/rem_num
        # Also d >= min_d
        lower_bound = max(min_d, ceil_div(rem_den, rem_num))
        
        d = lower_bound
        
        # Pre-calculate float rem for faster bounds
        rem_float = rem_num / rem_den
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > MAX_DENOM:
                break
            
            # Optimization: Use float approximation for max sum bound
            # max sum of depth terms starting from d is approx sum(1/(d+i))
            max_sum_float = 0.0
            for i in range(depth):
                max_sum_float += 1.0 / (d + i)
                
            # If even the max possible sum (using smallest denominators) is less than rem, 
            # we can't reach rem.
            if max_sum_float < rem_float - 1e-12:
                break

            # Pick d
            # new_rem = rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num < 0:
                d += 1
                continue
                
            if new_num == 0:
                if depth == 1:
                    return path + [d]
                else:
                    d += 1
                    continue

            # Reduce the fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g

            # Optimization: Check if remaining depth-1 terms can sum to new_rem
            # If not, and since increasing d makes new_rem larger and max sum smaller, we can break.
            if depth > 1:
                max_next_sum = 0.0
                for i in range(depth - 1):
                    max_next_sum += 1.0 / (d + 1 + i)
                
                if max_next_sum < (new_num / new_den) - 1e-12:
                    break

            new_min_d = d + 1
            new_depth = depth - 1
            new_path = path + [d]
            
            res = dfs(new_num, new_den, new_min_d, new_depth, new_path)
            if res is not None:
                return res
            
            d += 1
            
        memo.add(state)
        return None

    # Search for minimal k
    # We determined k=7 is impossible (max sum < 1).
    # Start from k=8.
    
    for k in range(8, 257):
        # Quick check if max possible sum < 1 using floats
        max_sum_float = sum(1.0 / (MIN_DENOM + i) for i in range(k))
        if max_sum_float < 1.0:
            continue

        result = dfs(1, 1, MIN_DENOM, k, [])
        if result is not None:
            return result
            
    # Fallback: Greedy algorithm using integer arithmetic
    def greedy(rem_num, rem_den, min_d):
        path = []
        while rem_num > 0:
            if time.time() > deadline:
                return None
            # Find smallest d >= min_d such that 1/d <= rem_num/rem_den
            d = max(min_d, ceil_div(rem_den, rem_num))
            if d > MAX_DENOM:
                return None
            path.append(d)
            # new_rem = rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            rem_num = new_num
            rem_den = new_den
            min_d = d + 1
        return path

    return greedy(1, 1, MIN_DENOM)
