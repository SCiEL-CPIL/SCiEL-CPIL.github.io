"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7] 
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 85
    
    def search(k, min_d, remaining, used):
        """
        Find k distinct denominators >= min_d summing to `remaining`.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            if remaining > 0:
                d = 1 / remaining
                if d >= min_d and d >= 5 and abs(d - round(d)) < 1e-9:
                    d_int = int(round(d))
                    if d_int not in used and d_int <= 10000000:
                        return [d_int]
            return None
        
        # Pruning: Max possible sum with k terms starting from min_d
        # is sum(1/(min_d + i) for i in range(k))
        max_possible = sum(Fraction(1, min_d + i) for i in range(k))
        if max_possible < remaining:
            return None
        
        # Lower bound for next denominator
        # 1/d < remaining/k is not quite right.
        # We need 1/d1 + ... + 1/dk = remaining.
        # 1/d1 < remaining.
        # Also 1/d1 + (k-1)/d1 > remaining is not a bound.
        # Max sum of remaining k-1 terms is sum(1/(d1+1+i)).
        # So 1/d1 + sum(1/(d1+1+i) for i in range(k-1)) >= remaining.
        
        # Simple lower bound: d1 must be such that 1/d1 < remaining.
        # So d1 > 1/remaining.
        lower_bound = max(min_d, int(1 / remaining) + 1)
        
        # Upper bound:
        # 1/d1 must be large enough so that remaining (k-1) terms can sum to remaining - 1/d1.
        # The max sum of k-1 terms starting from d1+1 is sum(1/(d1+1+i) for i in range(k-1)).
        # If remaining - 1/d1 > max_rest, then d1 is too large (1/d1 too small).
        # So we iterate d1 from lower_bound upwards.
        
        # Also, 1/d1 < remaining.
        
        # To limit the loop, we can find an upper bound for d1.
        # If 1/d1 + (k-1)/(d1+1) < remaining, then even the max possible sum of the rest is too small?
        # No, that's the condition for d1 being TOO LARGE.
        # Wait. If d1 is large, 1/d1 is small.
        # We need 1/d1 + Rest = remaining.
        # Rest <= sum(1/(d1+1+i)).
        # So 1/d1 + sum(1/(dd1+1+i)) >= remaining.
        # This inequality holds for small d1.
        # As d1 increases, LHS decreases.
        # So there is a max d1.
        
        # Let's just iterate and break if condition fails.
        
        for d1 in range(lower_bound, 10000001):
            if time.time() > deadline:
                return None
            
            if d1 in used:
                continue
            
            frac = Fraction(1, d1)
            if frac >= remaining:
                break # Since d1 is increasing, 1/d1 is decreasing. 
                       # If 1/d1 >= remaining, we skipped too many? 
                       # No, we need 1/d1 < remaining. 
                       # If 1/d1 >= remaining, then remaining - 1/d1 <= 0. 
                       # But we need positive terms. So 1/d1 MUST be < remaining.
                       # So if 1/d1 >= remaining, we should have started at a larger d1?
                       # No, lower_bound ensures 1/d1 < remaining.
                       # So this shouldn't happen if lower_bound is correct.
                       # But if lower_bound is 1/remaining + 1, then 1/d1 < remaining.
                       
            new_remaining = remaining - frac
            
            # Check if new_remaining can be formed by k-1 terms > d1
            if k - 1 > 0:
                max_rest = sum(Fraction(1, d1 + 1 + i) for i in range(k - 1))
                if max_rest < new_remaining:
                    # d1 is too large, 1/d1 is too small, leaving too much for the rest
                    # which can't be covered by the max possible sum of the rest.
                    # Wait. If max_rest < new_remaining, it means the rest is TOO SMALL.
                    # So we need a LARGER 1/d1, i.e., SMALLER d1.
                    # But we are iterating d1 upwards.
                    # So if this happens, we should have found a solution with a smaller d1.
                    # If we didn't, then no solution exists for this k?
                    # No, we iterate d1 from lower_bound up.
                    # If d1 is small, 1/d1 is large, new_remaining is small.
                    # max_rest is large. So max_rest >= new_remaining is likely true.
                    # As d1 increases, new_remaining increases.
                    # Eventually new_remaining > max_rest.
                    # Then we break? No, we continue?
                    # If new_remaining > max_rest, it means we can't form the sum.
                    # And for larger d1, new_remaining will be even larger, max_rest smaller.
                    # So we can break.
                    break
            
            result = search(k - 1, d1 + 1, new_remaining, used | {d1})
            if result is not None:
                return [d1] + result
        
        return None

    # Try k = 4, 5, 6, 7, 8
    for k in range(4, 10):
        if time.time() > deadline:
            break
        result = search(k, 5, Fraction(1, 1), set())
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == k and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback to DP if DFS fails or times out (shouldn't happen for small k)
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
