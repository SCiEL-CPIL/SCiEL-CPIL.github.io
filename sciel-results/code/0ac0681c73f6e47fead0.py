"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Fallback solution generator (Greedy)
    def greedy_solution():
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
            if time.time() > deadline:
                break
        return values

    best_solution = greedy_solution()
    best_max_den = max(best_solution)

    # DFS to find a solution with max denominator <= M
    # Uses integer arithmetic for speed where possible
    def dfs(rem_num, rem_den, terms_left, min_den, max_den, current):
        if terms_left == 0:
            return [] if rem_num == 0 else None
        
        if terms_left == 1:
            if rem_num != 1:
                return None
            d = rem_den
            if min_den <= d <= max_den:
                return [d]
            return None

        # Pruning:
        # Max sum of terms_left terms from [min_den, max_den]
        # Approx: terms_left / min_den
        # Min sum of terms_left terms from [min_den, max_den]
        # Approx: terms_left / max_den
        
        # Tighter bounds:
        # Max possible sum: 1/min_den + 1/(min_den+1) + ...
        # Min possible sum: 1/max_den + 1/(max_den-1) + ...
        
        # Quick float check
        max_possible = sum(1.0 / (min_den + i) for i in range(terms_left))
        min_possible = sum(1.0 / (max_den - i) for i in range(terms_left))
        
        target_val = rem_num / rem_den
        
        if target_val > max_possible:
            return None
        if target_val < min_possible:
            return None
            
        # d_min: 1/d <= target_val => d >= 1/target_val
        d_min = max(min_den, int(1.0 / target_val) + 1)
        if d_min < min_den:
            d_min = min_den
            
        # d_max: 
        # We need remaining sum after 1/d to be >= min sum of rest.
        # target_val - 1/d >= (terms_left - 1) / max_den
        # 1/d <= target_val - (terms_left - 1) / max_den
        # d >= 1 / (target_val - (terms_left - 1) / max_den)
        # This is a lower bound.
        
        # Also target_val - 1/d <= max sum of rest
        # target_val - 1/d <= (terms_left - 1) / (d + 1) approx
        # target_val <= 1/d + (terms_left - 1) / (d + 1)
        # For large d, target_val <= terms_left / d
        # d <= terms_left / target_val
        d_max = min(max_den, int(terms_left / target_val))
        
        if d_max < d_min:
            return None
            
        for d in range(d_min, d_max + 1):
            # Exact check: 1/d <= target_val
            if rem_num * d < rem_den: # 1/d < target
                # 1/d must be strictly less than or equal? 
                # Actually 1/d can be equal if terms_left==1, handled above.
                # If 1/d > target_val, skip.
                # rem_num/d > rem_den => rem_num * 1 > rem_den * d => rem_num > rem_den * d
                # Here we check if 1/d < target_val?
                # No, we need 1/d <= target_val.
                # If rem_num * d > rem_den, then 1/d < rem_num/rem_den. OK.
                # If rem_num * d == rem_den, 1/d == target. Only valid if terms_left==1.
                # If rem_num * d < rem_den, 1/d > target. Invalid.
                continue
            
            # Calculate new remaining
            # rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num <= 0:
                continue
                
            # Reduce fraction
            g = __import__('math').gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            result = dfs(new_num, new_den, terms_left - 1, d + 1, max_den, current + [d])
            if result is not None:
                return [d] + result
        return None

    # Try to find a solution with smaller max denominator
    # We iterate M upwards. If we find a solution with max denom <= M, we update best.
    # Since we want to MINIMIZE the max denom, we should try small M first.
    # But we already have a fallback.
    # Let's try M = 100, 200, 500, 1000...
    
    m_bounds = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for M in m_bounds:
        if time.time() > deadline:
            break
        if M >= best_max_den:
            break
            
        # Try different k values
        # k=3 is fast, k=4 slower, etc.
        for k in range(3, 8):
            if time.time() > deadline:
                break
            # Quick check: min possible sum with k terms from [1, M]
            # If 5/31 > sum(1/1..1/k) then impossible? No, min_den is 1.
            # Max sum with k terms from [1, M] is 1/1 + 1/2 + ...
            # Min sum with k terms from [1, M] is 1/M + 1/(M-1) + ...
            # If 5/31 < min_sum, impossible.
            
            sol = dfs(target.numerator, target.denominator, k, 1, M, [])
            if sol:
                max_d = max(sol)
                if max_d < best_max_den:
                    best_max_den = max_d
                    best_solution = sol
                break # Found one for this M and k, try next M? 
                       # Actually, if we found one with max_d <= M, and we are increasing M,
                       # the next M might allow a solution with even smaller max_d?
                       # No, we are searching for ANY solution with max <= M.
                       # If we find one, its max is <= M.
                       # We want the global min max.
                       # So we should continue checking smaller M? 
                       # We are iterating M upwards. So the first M we find a solution for
                       # gives a candidate. But a smaller M might have a solution too?
                       # We started from small M. So if we found a solution at M=100,
                       # we didn't find one at M=50. So 100 is the min for this k.
                       # But a larger k might work with M=50.
                       # So we should not break out of the k loop immediately if we want to check other k for this M?
                       # Actually, if we found a solution at M=100, we update best_max_den to <= 100.
                       # Then for M=200, we only care if we can find something < 100.
                       # But we already know M=50 had no solution for this k.
                       # So we can just break out of the k loop and go to next M?
                       # No, because for M=100, k=4 might fail but k=5 might succeed with max_d=90.
                       # So we should check all k for the current M.
                       # But if we find a solution with max_d=90 at M=100, we update best.
                       # Then for M=200, we look for max_d < 90.
                       # This logic holds.
    
    return best_solution

# EVOLVE-BLOCK-END
