from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin
    
    # Target: sum = 1, min denom = 5, distinct integers in [5, 10^7]
    # Minimize count k
    
    def dfs(remaining, min_denom, terms_left, current_path):
        if time.time() > deadline:
            return None
            
        if terms_left == 0:
            if remaining == 0:
                return list(current_path)
            else:
                return None
        
        if remaining == 0:
            return None  # Should not happen with terms_left > 0
            
        if remaining < 0:
            return None
        
        # Pruning: Max possible sum with 'terms_left' terms starting from min_denom
        # Sum_{i=0}^{terms_left-1} 1/(min_denom + i)
        max_sum = Fraction(0, 1)
        for i in range(terms_left):
            max_sum += Fraction(1, min_denom + i)
        if max_sum < remaining:
            return None
        
        # Lower bound for next denominator d
        # 1/d <= remaining  => d >= ceil(1/remaining)
        # But since denominators must be distinct and increasing, d >= min_denom
        # Also 1/d < remaining => d > 1/remaining
        # Actually we need 1/d <= remaining for feasibility (strictly less if more terms)
        # But we want sum of 1/d_i = remaining.
        # If 1/d > remaining, then d < 1/remaining, which is too small? No.
        # If 1/d > remaining, then we overshoot. So we need 1/d <= remaining.
        # d >= 1/remaining.
        
        # Let's compute d_min
        d_min = min_denom
        if remaining > 0:
            # d > 1/remaining is not strictly required, 1/d can be <= remaining
            # But if 1/d > remaining, we can't subtract it.
            # So we need 1/d <= remaining => d >= 1/remaining.
            # Since d is integer, d >= ceil(1/remaining).
            inv_rem = Fraction(1, 1) / remaining
            # ceil(1/remaining)
            d_min_candidate = int(inv_rem)
            if Fraction(d_min_candidate, 1) < inv_rem:
                d_min_candidate += 1
            if d_min_candidate > d_min:
                d_min = d_min_candidate
        
        # Upper bound for d
        # We need to pick 'terms_left' denominators d, d2, ..., d_k
        # The smallest possible sum for the remaining 'terms_left-1' terms (after d)
        # is achieved by large denominators, but we are bounded by the *max* sum of the *subsequent* terms.
        # Actually, we need the *max* sum of the remaining 'terms_left-1' terms to be at least remaining - 1/d.
        # The max sum of 'terms_left-1' distinct denominators starting from d+1 is:
        # Sum_{i=1}^{terms_left-1} 1/(d+i)
        # We need this >= remaining - 1/d.
        
        d = d_min
        while True:
            # Check if we can finish with this d
            rem_after = remaining - Fraction(1, d)
            if rem_after < 0:
                # This means 1/d > remaining, so we should have started at a larger d?
                # No, d_min was calculated as ceil(1/remaining), so 1/d_min <= remaining.
                # So rem_after >= 0.
                break
                
            if rem_after == 0:
                if terms_left == 1:
                    current_path.append(d)
                    return list(current_path)
                else:
                    # We have 0 remaining sum but terms left. Impossible.
                    pass
            
            # Check max sum of remaining terms
            if terms_left > 1:
                max_sub_sum = Fraction(0, 1)
                for i in range(1, terms_left):
                    max_sub_sum += Fraction(1, d + i)
                if max_sub_sum < rem_after:
                    # Even with the smallest possible next denominators (d+1, d+2...), 
                    # we can't reach the remaining sum.
                    # Since increasing d increases the denominators d+i, the max sum decreases.
                    # So for larger d, it will be even harder.
                    break
            
            # Recurse
            current_path.append(d)
            res = dfs(rem_after, d + 1, terms_left - 1, current_path)
            current_path.pop()
            
            if res:
                return res
            
            # Increment d
            d += 1
            # Optimization: if d becomes very large, we might timeout.
            # But the break condition above should catch it.
            
        return None

    # Try k from 5 to 12
    for k in range(5, 13):
        # Initial call
        # remaining = 1
        # min_denom = 5
        # terms_left = k
        res = dfs(Fraction(1, 1), 5, k, [])
        if res:
            return res
            
    # Fallback: return a known valid solution if search fails (should not happen)
    # Example: 5, 6, 7, 42, 140, ... (Need to verify)
    # Let's just return a large set from the original logic if this fails, 
    # but the search should find k=5 or 6.
    raise RuntimeError("No solution found within time limit")
