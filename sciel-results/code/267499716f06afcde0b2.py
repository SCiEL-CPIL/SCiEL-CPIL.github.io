import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit
    
    for k in range(5, 257):
        if time.time() > deadline:
            break
        result = search_egyptian(k, 5)
        if result and result != "TIMEOUT":
            return result
            
    raise RuntimeError("no solution found")

def search_egyptian(k, min_denom):
    target = Fraction(1, 1)
    
    def dfs(remaining_k, current_sum, last_denom, current_list):
        if remaining_k == 0:
            if current_sum == target:
                return list(current_list)
            return None
        
        rem = target - current_sum
        if rem <= 0:
            return None
            
        # Pruning: Max possible sum
        if current_sum + Fraction(remaining_k, last_denom + 1) < target:
            return None

        min_d = last_denom + 1
        # d > 1/rem
        # 1/d < rem => d > 1/rem
        # If rem is Fraction, 1/rem is Fraction.
        # We need integer d.
        # d_min_candidate = int(1/rem) + 1
        # But also d >= min_d
        
        # Upper bound:
        # rem - 1/d <= (remaining_k - 1) / (d + 1)
        # This inequality determines the max d.
        # For large d, LHS ~ rem, RHS ~ 0.
        # So d cannot be too large.
        
        # Estimate max_d
        # rem - 1/d <= (k-1)/(d+1)
        # rem <= 1/d + (k-1)/(d+1)
        # For large d, RHS ~ k/d.
        # So d <= k/rem.
        
        max_d = int(remaining_k / rem) + 2
        if max_d < min_d:
            max_d = min_d

        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return "TIMEOUT"
            
            # Check 1/d < rem
            if Fraction(1, d) >= rem:
                continue
                
            # Check if we can fill the rest
            if remaining_k > 1:
                if rem - Fraction(1, d) > Fraction(remaining_k - 1, d + 1):
                    break

            current_list.append(d)
            res = dfs(remaining_k - 1, current_sum + Fraction(1, d), d, current_list)
            current_list.pop()
            if res:
                return res
                
        return None

    return dfs(k, Fraction(0, 1), min_denom - 1, [])
