import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    target = Fraction(5, 31)
    max_terms = 256
    
    # Helper to check if a solution exists for a given max denominator K
    def feasible(K):
        # Pruning pre-calculations
        # min_sum[rem_terms] = sum of 1/(K-j) for j=0..rem_terms-1
        # This is the minimum possible sum of 'rem_terms' distinct denominators <= K
        min_sum = [Fraction(0, 1)] * (max_terms + 1)
        s = Fraction(0, 1)
        for i in range(1, max_terms + 1):
            s += Fraction(1, K - i + 1)
            min_sum[i] = s
            
        # max_sum_from[d][rem_terms] = sum of 1/(d+j) for j=1..rem_terms
        # We can compute this on the fly or precompute for d in range
        # Given K is small, d is small.
        
        path = []
        
        def dfs(rem, last_d, count):
            if time.time() > deadline:
                return False
            
            if rem == 0:
                return True
            
            if count >= max_terms:
                return False
                
            # Lower bound for next denominator
            # 1/d <= rem => d >= 1/rem
            # Also d > last_d
            d_min = max(last_d + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            # Upper bound for next denominator
            # We must be able to form 'rem - 1/d' with 'max_terms - count - 1' terms
            # Max sum of those terms is max_sum_from[d][remaining]
            # Min sum of those terms is min_sum[remaining]
            
            remaining = max_terms - count - 1
            
            for d in range(d_min, K + 1):
                if time.time() > deadline:
                    return False
                
                # 1/d must be <= rem
                if Fraction(1, d) > rem:
                    continue
                
                new_rem = rem - Fraction(1, d)
                
                if new_rem == 0:
                    path.append(d)
                    return True
                    
                if remaining == 0:
                    continue
                    
                # Pruning 1: new_rem must be >= min_sum[remaining]
                # min_sum[remaining] is sum of 1/K + 1/(K-1) + ...
                if new_rem < min_sum[remaining]:
                    # As d increases, new_rem increases.
                    # So we might reach a valid range.
                    continue
                
                # Pruning 2: new_rem must be <= max possible sum of 'remaining' terms > d
                # max_sum = sum(1/(d+j) for j in 1..remaining)
                # If new_rem > max_sum, then we need smaller denominators for the rest,
                # which means we need a smaller 'd' to allow smaller 'd+j'.
                # But we are iterating d upwards.
                # So if new_rem > max_sum(d), it will be even worse for larger d.
                # Thus, we can break.
                
                max_sum_rest = Fraction(0, 1)
                for j in range(1, remaining + 1):
                    max_sum_rest += Fraction(1, d + j)
                    
                if new_rem > max_sum_rest:
                    break
                
                path.append(d)
                if dfs(new_rem, d, count + 1):
                    return True
                path.pop()
                
            return False
        
        if dfs(target, 0, 0):
            return True, path[:]
        return False, []

    # Binary search for the minimum K
    low = 1
    high = 1736  # From seed analysis
    best_path = None
    
    while low <= high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        ok, path = feasible(mid)
        
        if ok:
            best_path = path
            high = mid - 1
        else:
            low = mid + 1
            
    if best_path:
        return best_path
    else:
        # Fallback to seed solution if search fails
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

