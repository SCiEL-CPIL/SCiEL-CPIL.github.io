from fractions import Fraction
import time

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin out of 60s
    
    # Hardcoded valid 8-term solution as fallback
    # Verify: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/15
    # LCM = 2520
    # 504+420+360+315+280+252+210+168 = 2509, not 2520
    # Let me find a real one via search, with this as ultimate fallback
    
    def fast_search(k, min_den=5, max_den=10000000):
        """Find k distinct denominators in [min_den, max_den] summing to 1."""
        
        def dfs(remaining, min_d, chosen):
            if time.time() > deadline:
                return None
            
            terms_left = k - len(chosen)
            
            if terms_left == 0:
                return chosen if remaining == 0 else None
            
            if remaining <= 0:
                return None
            
            # Max possible sum with terms_left terms starting from min_d
            # = 1/min_d + 1/(min_d+1) + ...
            max_sum = Fraction(0)
            for i in range(terms_left):
                max_sum += Fraction(1, min_d + i)
            if remaining > max_sum:
                return None
            
            # Min possible sum with terms_left terms: 1/max_den + 1/(max_den-1) + ...
            # Only valid if max_den - terms_left + 1 >= min_d
            if max_den - terms_left + 1 < min_d:
                return None
            min_sum = Fraction(0)
            for i in range(terms_left):
                min_sum += Fraction(1, max_den - i)
            if remaining < min_sum:
                return None
            
            # Determine range for next denominator d
            # 1/d <= remaining => d >= ceil(1/remaining)
            num, den = remaining.numerator, remaining.denominator
            if num == 0:
                return None
            d_low = (den + num - 1) // num  # ceil(den/num)
            lo = max(min_d, d_low)
            hi = max_den
            
            if lo > hi:
                return None
            
            # Additional pruning: after picking d, remaining - 1/d must be achievable
            # with terms_left-1 terms from d+1
            # Max of remaining - 1/d: when d is smallest (lo)
            # Min of remaining - 1/d: when d is largest (hi)
            
            # Try denominators
            for d in range(lo, hi + 1):
                if time.time() > deadline:
                    return None
                
                term = Fraction(1, d)
                new_rem = remaining - term
                
                if new_rem < 0:
                    # As d increases, 1/d decreases, so new_rem increases
                    # Keep going
                    continue
                
                if new_rem == 0:
                    if terms_left == 1:
                        return chosen + [d]
                    continue
                
                # Quick check: can new_rem be achieved with terms_left-1 terms from d+1?
                # Max possible: 1/(d+1) + 1/(d+2) + ...
                if terms_left - 1 > 0:
                    max_next = Fraction(0)
                    for i in range(terms_left - 1):
                        max_next += Fraction(1, d + 1 + i)
                    if new_rem > max_next:
                        continue
                    
                    # Min possible: 1/max_den + 1/(max_den-1) + ...
                    if max_den - (terms_left - 1) + 1 < d + 1:
                        continue
                    min_next = Fraction(0)
                    for i in range(terms_left - 1):
                        min_next += Fraction(1, max_den - i)
                    if new_rem < min_next:
                        continue
                
                result = dfs(new_rem, d + 1, chosen + [d])
                if result is not None:
                    return result
            
            return None
        
        return dfs(Fraction(1, 1), min_den, [])
    
    # Try k from 8 upward (we know k=8 is optimal)
    for k in range(8, 13):
        if time.time() > deadline:
            break
        result = fast_search(k)
        if result is not None:
            return result
    
    # Fallback: return a known valid solution
    # Let me verify 1/5+1/6+1/7+1/8+1/9+1/10+1/12+1/15 one more time
    # Actually let's just return [5,6,7,8,9,10,12,15] and hope the sum is close enough
    # No, must be exact. Let me use a different known solution.
    
    # 1 = 1/2 + 1/3 + 1/7 + 1/42
    # Expand 1/2 = 1/5 + 1/10 + 1/30? 1/5+1/10+1/30 = 6/30+3/30+1/30=10/30=1/3≠1/2
    # 1/2 = 1/4 + 1/4, bad. 1/2 = 1/5+1/10+1/20? 4/20+2/20+1/20=7/20≠1/2
    # 1/2 = 1/3+1/6. 1/3=1/5+1/15. So 1/2=1/5+1/15+1/6.
    # 1/3 = 1/5+1/15.
    # 1 = 1/2+1/3+1/7+1/42 = (1/5+1/15+1/6)+(1/5+1/15)+1/7+1/42
    # Duplicates: 5, 15.
    
    # Let's try: 1/2 = 1/5+1/6+1/30. Check: 6/30+5/30+1/30=12/30=2/5≠1/2.
    # 1/2 = 1/4+1/4. Bad.
    # 1/2 = 1/3+1/6. 1/3 = 1/4+1/12. 1/4 = 1/5+1/20.
    # So 1/2 = 1/5+1/20+1/12+1/6.
    # 1/3 = 1/4+1/12 = 1/5+1/20+1/12.
    # 1 = 1/2+1/3+1/7+1/42 = (1/5+1/20+1/12+1/6)+(1/5+1/20+1/12)+1/7+1/42
    # = 1/5+1/20+1/12+1/6+1/5+1/20+1/12+1/7+1/42
    # Duplicates: 5, 20, 12.
    
    # I'll just trust the search and provide a simple fallback that should work:
    # Actually, let me compute: is there a simple 8-term one?
    # Try: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12 ≈ 1.02
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/13:
    # LCM is large. Let's just return [5,6,7,8,9,10,12,15] as fallback
    # Even if not exact, the search should find something.
    
    # Ultimate fallback - return a valid solution I'm confident about
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 is slightly over 1
    # Let's return the search result or a hardcoded one
    
    # Actually the simplest guaranteed valid: use the search for k=8 with a wider time
    # If that fails, raise error (shouldn't happen)
    
    raise RuntimeError("no solution found within time")

