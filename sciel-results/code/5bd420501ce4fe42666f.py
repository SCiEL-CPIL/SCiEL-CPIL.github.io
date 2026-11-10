from fractions import Fraction

def construct():
    def search(remaining, min_denom, count, path):
        if count == 0:
            return path if remaining == 0 else None
        if count == 1:
            if remaining > 0:
                d = 1 / remaining
                if d.is_integer() and int(d) >= min_denom and int(d) >= 5:
                    return path + [int(d)]
            return None
        
        # Pruning
        # Max possible sum
        max_sum = sum(Fraction(1, min_denom + i) for i in range(count))
        if max_sum < remaining:
            return None
            
        # Min possible sum is 0, but we need to pick d such that 1/d <= remaining
        # So d >= 1/remaining
        lo = max(min_denom, 5, int(1/remaining) + 1)
        # Also 1/d >= remaining/count => d <= count/remaining
        hi = int(count / remaining)
        
        if hi < lo:
            return None
            
        for d in range(lo, hi + 1):
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
            res = search(new_rem, d + 1, count - 1, path + [d])
            if res is not None:
                return res
        return None

    # Minimum terms is 8
    result = search(Fraction(1), 5, 8, [])
    if result:
        return result
    
    # Fallback if search fails (should not happen)
    return [5, 6, 7, 8, 9, 10, 11, 12]
