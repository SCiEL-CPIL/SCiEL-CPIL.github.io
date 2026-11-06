import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Target is to find k=7 solution
    # If found, score is 1/7.
    # If not, fallback to k=8, score 1/8.
    
    result = _search_k(7, deadline)
    if result:
        return result
    
    result = _search_k(8, deadline)
    if result:
        return result
        
    # Fallback to known valid k=8 if search fails (should not happen)
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/12 + 1/14 + 1/35 + 1/420? No.
    # A known k=8 solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/15 + 1/30?
    # Sum: 1/5+1/6+1/7+1/8+1/9+1/10+1/15+1/30
    # LCM 2520.
    # 504+420+360+315+280+252+168+84 = 2383 != 2520.
    
    # Just return the result from search.
    raise RuntimeError("no valid representation found")

def _search_k(k, deadline):
    """Search for k distinct denominators >= 5 summing to 1"""
    d = []
    
    def dfs(idx, cur_num, cur_den, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if cur_num == cur_den:
                return d[:]
            return None
        
        remaining_terms = k - idx
        rem_num = cur_den - cur_num
        rem_den = cur_den
        
        if rem_num <= 0:
            return None
        
        # Prune: max possible sum of remaining terms
        # Using float for speed, with safety margin
        max_possible_float = 0.0
        for i in range(remaining_terms):
            max_possible_float += 1.0 / (last_denom + 1 + i)
            
        if rem_num / rem_den > max_possible_float + 1e-12:
            return None
            
        # Prune: min possible sum of remaining terms
        # The smallest possible sum is if we pick the largest possible denominators?
        # Actually, we just need to ensure we don't overshoot.
        
        if remaining_terms == 1:
            if rem_den % rem_num == 0:
                d_val = rem_den // rem_num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, cur_den, cur_den, d_val):
                        return d[:]
                    d.pop()
            return None

        # Determine range for next denominator
        # 1/d <= rem_num/rem_den => d >= rem_den/rem_num
        min_d_float = rem_den / rem_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Upper bound:
        # 1/d >= rem_num/rem_den - sum of max possible remaining (terms-1)
        # Simpler upper bound: d <= rem_den / rem_num is wrong direction?
        # If we pick d too large, 1/d is too small, and we can't reach the sum with remaining terms?
        # Actually, if d is too large, the remaining sum required becomes larger?
        # No, if we pick 1/d, the new remainder is rem - 1/d.
        # We need rem - 1/d >= sum of (terms-1) smallest possible fractions?
        # The smallest possible fractions are those with largest denominators.
        # But we have an upper bound on denominators? No, just distinct and > last.
        # However, we must be able to form the remainder with (terms-1) terms.
        # The maximum sum of (terms-1) terms with denominators > d is approx (terms-1)/(d+1).
        # So we need: rem_num/rem_den - 1/d >= (terms-1)/(d+1) (approx)
        # This gives a lower bound on d? No, this is a constraint on d.
        # Let's use a simpler heuristic upper bound:
        # d <= rem_den / rem_num is the minimum d.
        # A common upper bound is d <= (rem_den * remaining_terms) / rem_num ? No.
        # Let's use: d <= rem_den / (rem_num - (remaining_terms-1) * something)
        
        # A robust upper bound:
        # If we choose d, the remaining sum is (rem_num*d - rem_den) / (rem_den*d).
        # This must be representable by (remaining_terms-1) terms > d.
        # Max sum of (remaining_terms-1) terms > d is approx (remaining_terms-1)/(d+1).
        # So: (rem_num*d - rem_den) / (rem_den*d) <= (remaining_terms-1)/(d+1)
        # (rem_num*d - rem_den) * (d+1) <= rem_den * d * (remaining_terms-1)
        # rem_num*d^2 + rem_num*d - rem_den*d - rem_den <= rem_den*d*(remaining_terms-1)
        # rem_num*d^2 + d*(rem_num - rem_den - rem_den*(remaining_terms-1)) - rem_den <= 0
        # This is a quadratic in d.
        
        # For simplicity and speed, we can just iterate d from min_d to a reasonable max_d.
        # A safe max_d is often rem_den // rem_num * remaining_terms + 100.
        max_d = int(rem_den / rem_num * remaining_terms) + 100
        
        if min_d > max_d:
            return None
            
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            # new fraction = cur_num/cur_den + 1/d_val
            new_num = cur_num * d_val + rem_den # Wait, cur_num/cur_den + 1/d_val = (cur_num*d_val + cur_den) / (cur_den*d_val)
            # But we are tracking remainder.
            # rem = rem_num/rem_den.
            # new_rem = rem_num/rem_den - 1/d_val = (rem_num*d_val - rem_den) / (rem_den*d_val)
            
            num_diff = rem_num * d_val - rem_den
            if num_diff < 0:
                # 1/d_val > rem, so we overshot. Since d_val is increasing, we can break?
                # No, 1/d_val decreases as d_val increases.
                # If 1/d_val > rem, then for smaller d_val it would be even larger.
                # But we are iterating d_val from min_d upwards.
                # min_d ensures 1/min_d <= rem. So num_diff should be >= 0.
                continue
                
            new_den = rem_den * d_val
            
            # Reduce
            g = gcd(num_diff, new_den)
            new_num = num_diff // g
            new_den = new_den // g
            
            # Check if new_num/new_den is feasible
            if new_num == 0:
                if remaining_terms - 1 == 0:
                    # This would mean we reached exactly 1, but we have terms left?
                    # No, if num_diff is 0, rem became 0.
                    # But we need to use all k terms?
                    # The problem says "Return 1..256 distinct integer denominators... whose reciprocal sum is exactly 1".
                    # It implies we use exactly the set we return.
                    # If we reach 0 remainder with terms left, we can't add more terms without exceeding 1.
                    # So this path is invalid unless remaining_terms-1 == 0.
                    pass 
                else:
                    continue
            elif new_num < 0:
                continue
            
            d.append(d_val)
            res = dfs(idx + 1, cur_num + new_num, cur_den * new_den // g, d_val) # This state tracking is getting complex.
            
            # Let's stick to the previous program's state tracking which was working for k=8.
            # It tracked cur_num/cur_den as the SUM so far.
            
            d.pop()
            
        return None

    # Re-implementing with the SUM state tracking as in the original, which was stable.
    def dfs_sum(idx, cur_num, cur_den, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if cur_num == cur_den:
                return d[:]
            return None
        
        remaining_terms = k - idx
        rem_num = cur_den - cur_num
        rem_den = cur_den
        
        if rem_num <= 0:
            return None
            
        # Prune: max possible sum
        max_possible_float = 0.0
        for i in range(remaining_terms):
            max_possible_float += 1.0 / (last_denom + 1 + i)
        if rem_num / rem_den > max_possible_float + 1e-12:
            return None

        if remaining_terms == 1:
            if rem_den % rem_num == 0:
                d_val = rem_den // rem_num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    d.append(d_val)
                    if dfs_sum(idx + 1, cur_den, cur_den, d_val):
                        return d[:]
                    d.pop()
            return None

        min_d_float = rem_den / rem_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Upper bound heuristic
        max_d = int(rem_den / rem_num * remaining_terms) + 100
        
        if min_d > max_d:
            return None
            
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            # new_sum = cur_num/cur_den + 1/d_val
            new_num = cur_num * d_val + cur_den
            new_den = cur_den * d_val
            
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            if new_num >= new_den:
                # Overshot 1. Since d_val increases, 1/d_val decreases, so new_sum decreases.
                # Wait.
                # If d_val is small, 1/d_val is large.
                # We start from min_d.
                # If new_sum >= 1 at min_d, then for larger d_val, new_sum will be smaller.
                # So we should NOT break, but continue?
                # Actually, if new_sum >= 1, it's invalid.
                # As d_val increases, new_sum decreases.
                # So we might find a valid d_val later.
                continue
            
            # Prune: can we reach 1 with remaining terms?
            rem2_num = new_den - new_num
            rem2_den = new_den
            rem2_float = rem2_num / rem2_den
            
            if remaining_terms - 1 > 0:
                max_rem_float = 0.0
                for i in range(remaining_terms - 1):
                    max_rem_float += 1.0 / (d_val + 1 + i)
                if rem2_float > max_rem_float + 1e-12:
                    continue
            
            d.append(d_val)
            res = dfs_sum(idx + 1, new_num, new_den, d_val)
            if res:
                return res
            d.pop()
        
        return None

    return dfs_sum(0, 0, 1, 4)

