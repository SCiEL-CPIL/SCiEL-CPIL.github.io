import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    def find_solution(k, max_denom=10000000):
        """Find k distinct denominators in [5, max_denom] with reciprocal sum = 1."""
        best = [None]
        target = Fraction(1, 1)
        
        def backtrack(start_idx, remaining_terms, current_sum, current_denoms):
            if time.time() > deadline:
                return
            
            if remaining_terms == 0:
                if current_sum == target:
                    best[0] = current_denoms[:]
                return
            
            rem = target - current_sum
            if rem <= 0:
                return
            
            # We need 1/d + ... = rem with remaining_terms terms, d >= start_idx
            # 1/d < rem => d > 1/rem
            # 1/d > rem / remaining_terms => d < remaining_terms / rem
            # So d in (1/rem, remaining_terms/rem), d >= start_idx
            
            # Compute bounds
            # d > 1/rem means d >= floor(1/rem) + 1
            # d < remaining_terms/rem means d <= ceil(remaining_terms/rem) - 1
            
            # Using Fraction arithmetic
            # 1/rem: if rem = p/q, then 1/rem = q/p
            # remaining_terms/rem = remaining_terms * q / p
            
            num, den = rem.numerator, rem.denominator
            # 1/rem = den/num
            # remaining_terms/rem = remaining_terms * den / num
            
            d_min = max(start_idx, den // num + 1)
            # d < remaining_terms * den / num
            # d <= (remaining_terms * den - 1) // num
            d_max = (remaining_terms * den - 1) // num
            d_max = min(d_max, max_denom)
            
            if d_min > d_max:
                return
            
            # Additional pruning: max possible sum with remaining terms
            # If we use d, d+1, ..., the max sum is 1/d + 1/(d+1) + ... 
            # But simpler: even if all remaining are 1/start_idx, can we reach rem?
            # rem <= remaining_terms / start_idx
            if rem > remaining_terms * Fraction(1, start_idx):
                return
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return
                
                new_sum = current_sum + Fraction(1, d)
                if new_sum > target:
                    # d is too small (1/d too large), try larger d
                    continue
                
                # Check if remaining can still sum to target - new_sum
                rem2 = target - new_sum
                if rem2 <= 0:
                    if rem2 == 0 and remaining_terms == 1:
                        current_denoms.append(d)
                        best[0] = current_denoms[:]
                        current_denoms.pop()
                        return
                    continue
                
                # Max possible from remaining_terms-1 terms starting at d+1
                if rem2 > (remaining_terms - 1) * Fraction(1, d + 1):
                    # Can't reach even with smallest possible denominators
                    # Since d is increasing, this will only get worse
                    break
                
                current_denoms.append(d)
                backtrack(d + 1, remaining_terms - 1, new_sum, current_denoms)
                current_denoms.pop()
                
                if best[0] is not None:
                    return
        
        backtrack(5, k, Fraction(0, 1), [])
        return best[0]
    
    # Try k = 2, 3, 4, 5, ...
    for k in range(2, 10):
        if time.time() > deadline - 5:
            break
        sol = find_solution(k)
        if sol is not None:
            if len(sol) == k and len(set(sol)) == k:
                if all(5 <= d <= 10000000 for d in sol):
                    s = sum(Fraction(1, d) for d in sol)
                    if s == 1:
                        return sol
    
    raise RuntimeError("no solution found")
