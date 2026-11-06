import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s limit to be safe

    def solve(k, target, min_den, current):
        # target: Fraction
        # k: number of terms to find
        # min_den: minimum denominator for the next term
        # current: list of denominators found so far
        
        if time.time() > deadline:
            return None

        if k == 0:
            if target == 0:
                return current
            return None

        if k == 1:
            if target.denominator == 1:
                d = target.numerator
                if d >= min_den:
                    return current + [d]
            return None

        # Pruning: Max sum
        # Max sum of k terms starting from min_den
        # Sum(1/(min_den+i) for i in range(k))
        # If this is less than target, impossible.
        max_sum = sum(Fraction(1, min_den + i) for i in range(k))
        if max_sum < target:
            return None

        # Iterate d from min_den
        # Upper bound for d:
        # 1/d <= target/k ? No.
        # 1/d + 1/(d+1) + ... + 1/(d+k-1) >= target
        # This is a lower bound on d.
        # Also 1/d < target => d > 1/target.
        # And 1/d + 1/(d+1) + ... >= target.
        
        # A simpler upper bound:
        # The smallest remaining sum is 0.
        # The largest term 1/d must be such that the remaining k-1 terms can sum to target - 1/d.
        # Max sum of k-1 terms starting from d+1 is approx (k-1)/(d+1).
        # So target - 1/d <= (k-1)/(d+1) is NOT required, but if target - 1/d is too small, it's fine.
        # If target - 1/d is too LARGE for the max sum of remaining terms, then d is too small?
        # No, if d is small, 1/d is large, target - 1/d is small.
        # If d is large, 1/d is small, target - 1/d is large.
        # So if d is too large, target - 1/d becomes large, and might exceed the max sum of remaining terms.
        # So there is an upper bound on d.
        
        # Let's find d_min and d_max.
        # d_min = min_den
        # For d_max:
        # We need target - 1/d <= MaxSum(k-1, d+1).
        # MaxSum(k-1, d+1) < (k-1)/(d+1).
        # So target - 1/d <= (k-1)/(d+1).
        # target <= 1/d + (k-1)/(d+1).
        # For large d, RHS approx k/d.
        # So target <= k/d => d <= k/target.
        
        d_min = min_den
        d_max = int(k / target)
        
        # Also d must be such that 1/d < target (if k>1, we can't have 1/d >= target)
        # Actually if 1/d >= target, then remaining sum <= 0, which is impossible for k-1 > 0 positive terms.
        # So 1/d < target => d > 1/target.
        d_min = max(d_min, int(1/target) + 1)
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            
            rem_target = target - Fraction(1, d)
            if rem_target < 0:
                break # d is too small, 1/d > target
                
            # Recurse
            res = solve(k - 1, rem_target, d + 1, current + [d])
            if res:
                return res
                
        return None

    # Try k=5
    for k in range(5, 10):
        res = solve(k, Fraction(1, 1), 5, [])
        if res:
            # Verify
            if len(set(res)) == len(res) and all(5 <= d <= 10000000 for d in res):
                s = sum(Fraction(1, d) for d in res)
                if s == 1:
                    return res
                    
    # Fallback to seed-like behavior if search fails
    return [5, 6, 7, 8, 9, 10, 11, 252]
