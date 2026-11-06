import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def dfs(k, min_den, remaining, path):
        if time.time() > deadline:
            return None
        if k == 0:
            return path[:] if remaining == 0 else None
        
        if k == 1:
            if remaining > 0:
                inv = 1 / remaining
                if inv.is_integer():
                    d_val = int(inv)
                    if d_val > min_den:
                        return path + [d_val]
            return None

        # Pruning
        if remaining > k * Fraction(1, min_den):
            return None
        if remaining < k * Fraction(1, min_den + k - 1):
            return None

        d = min_den + 1
        # Upper bound for d based on max sum of remaining terms
        # remaining - 1/d <= (k-1)/(d+1)
        # d(remaining - 1) <= k - 1 - remaining
        # d >= (k - 1 - remaining) / (remaining - 1)
        # Since remaining < 1, this is a lower bound, which is d > 1/remaining.
        # We iterate upwards.
        
        while d <= 10000000:
            if 1 / d >= remaining:
                break
            
            new_rem = remaining - Fraction(1, d)
            # Check if new_rem is possible with k-1 terms > d
            if new_rem <= Fraction(k - 1, d + 1):
                res = dfs(k - 1, d, new_rem, path + [d])
                if res:
                    return res
            
            d += 1
            
        return None

    # Try to find minimal k
    for k in range(8, 13):
        if time.time() > deadline - 5:
            break
        res = dfs(k, 4, Fraction(1, 1), [])
        if res:
            # Verify
            if len(set(res)) == len(res) and all(x >= 5 for x in res) and sum(Fraction(1, x) for x in res) == 1:
                return res

    # Fallback to known k=9 solution
    return [5, 6, 7, 8, 9, 10, 12, 18, 315]
