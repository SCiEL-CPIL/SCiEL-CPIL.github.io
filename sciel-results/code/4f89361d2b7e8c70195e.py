"""Rewritten program: find fewest distinct denominators >=5 with reciprocal sum 1."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    # Deadline guard: evaluator kills after 90s; keep safety margin.
    deadline = time.time() + 85.0
    
    # We want to minimize the count of distinct denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1.
    #
    # Known small solutions:
    # 2 terms: 1/5 + 1/x = 1 => 1/x = 4/5 => x=5/4 not integer. No 2-term solution.
    # 3 terms: 1/a + 1/b + 1/c = 1 with a<=b<=c, a>=5.
    #   Try a=5: 1/b+1/c = 4/5 => (b-5)(c-5)=25. Factors of 25: 1,5,25.
    #     b-5=1 => b=6, c-5=25 => c=30. Sum: 1/5+1/6+1/30 = 6/30+5/30+1/30=12/30=2/5 != 1.
    #     Wait, let me recompute. 1/5+1/6+1/30 = 6/30+5/30+1/30 = 12/30 = 2/5. Not 1.
    #   I made an error. Let me redo: 1/a+1/b+1/c=1.
    #   Multiply by abc: bc+ac+ab = abc.
    #   For a=5: 5b+5c+bc = 5bc => bc-5bc = -5b-5c => -4bc = -5(b+c) => 4bc = 5(b+c).
    #   4bc - 5b - 5c = 0. Multiply by 4: 16bc - 20b - 20c = 0. Add 25: (4b-5)(4c-5) = 25.
    #   Factors of 25: 1,5,25.
    #     4b-5=1 => b=1.5 no.
    #     4b-5=5 => b=2.5 no.
    #     4b-5=25 => b=7.5 no.
    #   So no 3-term solution with a=5.
    #   
    #   Let me try a=6: 1/6+1/b+1/c=1 => 1/b+1/c=5/6.
    #   6b+6c+bc=6bc => 5bc-6b-6c=0. (5b-6)(5c-6)=36.
    #   Factors of 36: 1,2,3,4,6,9,12,18,36.
    #     5b-6=1 => b=7/5 no.
    #     5b-6=2 => b=8/5 no.
    #     5b-6=3 => b=9/5 no.
    #     5b-6=4 => b=2 no (<6, and also <5).
    #     5b-6=6 => b=12/5 no.
    #     5b-6=9 => b=3 no.
    #     5b-6=12 => b=18/5 no.
    #     5b-6=18 => b=24/5 no.
    #     5b-6=36 => b=42/5 no.
    #   No 3-term with a=6.
    #
    #   Actually, the known smallest Egyptian fraction for 1 with min denominator >= 5:
    #   Let me just search computationally.
    #
    # 4-term solutions: 1/a+1/b+1/c+1/d=1.
    #   Try a=5,b=6: 1/5+1/6=11/30. Need 1/c+1/d=19/30.
    #   30c+30d+cd=19cd => 18cd-30c-30d=0. (18c-30)(18d-30)=900.
    #   This gets complicated. Let me just do a computational search.
    
    def find_min_terms(max_terms=8, max_denom=10000000):
        """Search for minimum number of terms using backtracking."""
        # Try increasing number of terms
        for num_terms in range(3, max_terms + 1):
            result = _backtrack_search(num_terms, 5, 0, max_denom, deadline)
            if result is not None:
                return result
        return None
    
    def _backtrack_search(terms_left, min_denom, current_sum_frac, max_denom, deadline):
        """
        Backtracking search for exact sum = 1.
        terms_left: how many more denominators to pick
        min_denom: minimum allowed denominator (to ensure distinctness and ordering)
        current_sum_frac: Fraction representing current partial sum
        """
        if time.time() > deadline:
            return None
        
        remaining = Fraction(1) - current_sum_frac
        if remaining < 0:
            return None
        
        if terms_left == 1:
            # Need 1/d = remaining, so d = 1/remaining
            if remaining == 0:
                return []
            num, den = remaining.numerator, remaining.denominator
            if den == 0:
                return None
            if num == 0:
                return None
            # d = den/num must be integer
            if den % num == 0:
                d = den // num
                if d >= min_denom and d <= max_denom:
                    return [d]
            return None
        
        if terms_left == 2:
            # Need 1/a + 1/b = remaining, with a >= min_denom, a < b
            # 1/a + 1/b = r => b = a/(a*r - 1) where r = remaining
            # r = num/den
            # b = a / (a*num/den - 1) = a*den / (a*num - den)
            num_r, den_r = remaining.numerator, remaining.denominator
            if num_r == 0:
                return None
            
            # a must satisfy: a*num_r - den_r > 0 => a > den_r/num_r
            # and a >= min_denom
            a_min = max(min_denom, den_r // num_r + 1)
            
            # Also need b > a, so 1/b < 1/a, so remaining - 1/a < 1/a => remaining < 2/a => a < 2/remaining
            # a < 2*den_r/num_r
            a_max = min(max_denom, (2 * den_r) // num_r)
            
            # Also, with terms_left=2, the max possible sum with a >= a_min is 2/a_min
            # So we need 2/a_min >= remaining => a_min <= 2/remaining
            if a_min > a_max:
                return None
            
            # Upper bound on a: since b > a, we have 1/a + 1/b < 2/a, so 2/a > remaining => a < 2/remaining
            # Lower bound: 1/a < remaining => a > 1/remaining
            # Also a < b implies a < 2/remaining
            
            for a in range(a_min, a_max + 1):
                if time.time() > deadline:
                    return None
                # b = a*den_r / (a*num_r - den_r)
                denom_b = a * num_r - den_r
                if denom_b <= 0:
                    continue
                if (a * den_r) % denom_b != 0:
                    continue
                b = (a * den_r) // denom_b
                if b <= a or b > max_denom:
                    continue
                # Found a solution
                return [a, b]
            
            return None
        
        # General case: terms_left >= 3
        # Choose next denominator d >= min_denom
        # 1/d < remaining (since we need positive terms after)
        # So d > 1/remaining
        # Also, max possible sum with terms_left terms each <= 1/min_denom is terms_left/min_denom
        # So we need terms_left/min_denom >= remaining => min_denom <= terms_left/remaining
        
        num_r, den_r = remaining.numerator, remaining.denominator
        if num_r == 0:
            return None
        
        d_min = max(min_denom, den_r // num_r + 1)
        # d must be such that 1/d < remaining, and we can still achieve the remaining sum
        # with (terms_left - 1) terms each >= 1/d (but actually each term is 1/x where x > d)
        # Max remaining sum after picking d: (terms_left-1)/d (if all subsequent terms were 1/d, but they must be > d)
        # Actually, the max sum of (terms_left-1) terms with denominators > d is (terms_left-1)/(d+1)
        # So we need (terms_left-1)/(d+1) >= remaining - 1/d
        # This gives an upper bound on d.
        
        # Simpler upper bound: d < terms_left/remaining (since 1/d must be less than remaining, and we need enough room)
        # Actually: 1/d + (terms_left-1)/(d+1) >= remaining is a necessary condition
        # Let's use: d < terms_left / remaining as a loose upper bound
        d_max = min(max_denom, (terms_left * den_r) // num_r)
        
        if d_min > d_max:
            return None
        
        # Pruning: if even with the smallest possible denominators we can't reach remaining, prune
        # Max possible sum with terms_left terms starting from d_min: 
        # 1/d_min + 1/(d_min+1) + ... 
        # For a loose check: terms_left / d_min >= remaining
        if terms_left / d_min < remaining:
            return None
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            
            new_sum = current_sum_frac + Fraction(1, d)
            if new_sum >= 1:
                # Since d is increasing, new_sum will only get smaller... wait no.
                # Actually 1/d is decreasing, so as d increases, new_sum decreases.
                # If new_sum >= 1, we might still be OK if it equals 1 exactly with 0 terms left,
                # but we have terms_left > 1, so we need new_sum < 1.
                # If new_sum >= 1, skip (and for larger d, new_sum will be smaller, so continue)
                # Wait, if new_sum > 1, we need to increase d. If new_sum == 1 and terms_left > 1, that's invalid.
                if new_sum > 1:
                    continue
                elif new_sum == 1:
                    # Can't add more positive terms
                    continue
                else:
                    # new_sum < 1, recurse
                    result = _backtrack_search(terms_left - 1, d + 1, new_sum, max_denom, deadline)
                    if result is not None:
                        return [d] + result
        
        return None
    
    # Start search
    result = find_min_terms(max_terms=8, max_denom=10000000)
    
    if result is not None:
        # Validate
        s = sum(Fraction(1, d) for d in result)
        if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
            return result
    
    # Fallback: return a known valid solution if search failed
    # 1/5 + 1/6 + 1/7 + 1/42 = 1? Let's check: 1/5+1/6+1/7+1/42
    # = 42/210 + 35/210 + 30/210 + 5/210 = 112/210 = 8/15 != 1
    # 
    # Known: 1/2+1/3+1/6=1 but 2,3 < 5.
    # Let's just return a valid 4-term or 5-term solution.
    # 1/5 + 1/5 is not allowed (distinct).
    # 
    # Try: 1/5+1/6+1/10+1/30 = 6/30+5/30+3/30+1/30 = 15/30 = 1/2. No.
    # 
    # Let me find one: 
    # 1/5 + 1/6 + 1/7 + 1/43 + ... this is getting complex.
    # 
    # Actually, let me just try a few known decompositions.
    # 1 = 1/5 + 4/5. 4/5 = 1/6 + 19/30. 19/30 = 1/7 + 53/210. 53/210 = 1/5 + ... no, 5 already used.
    # 
    # Let's just return a hardcoded valid solution as fallback.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + ...
    # This is too many. Let me just do a quick greedy.
    
    # Fallback: use a known valid set
    # 1/5 + 1/6 + 1/7 + 1/42 = 1/5+1/6+1/7+1/42
    # LCM(5,6,7,42) = 210
    # 42+35+30+5 = 112/210 = 8/15. Not 1.
    
    # Let me just return [5, 6, 7, 42, 140, 420, 840] and check... no, this is not efficient.
    
    # As a last resort, return a valid but possibly non-minimal solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # This sum is > 1, so we need to be careful.
    
    # Simplest fallback: return a hardcoded valid solution
    # After some thought, 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20+1/21+1/22+1/23+1/24+1/25+1/26+1/27+1/28+1/29+1/30
    # is way more than 1.
    
    # Let me just return a small valid set I can verify:
    # 1/5 + 1/6 + 1/7 + 1/43 + 1/210 + 1/420 + 1/840 ... no.
    
    # Fallback: return [5, 6, 7, 42, 140, 420, 840, 1680, 3360, 6720, 13440, 26880, 53760, 107520, 215040, 430080, 860160, 1720320, 3440640, 6881280]
    # This is a greedy Egyptian fraction for 1 starting from 5, but it's many terms.
    
    # Actually, the simplest valid fallback:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This is too many and sum > 1.
    
    # Let me just return a hardcoded valid 4-term solution if I can find one, else a longer one.
    # For now, return the result of a quick greedy to ensure validity.
    
    def greedy_egyptian(target, min_denom, max_terms=256):
        """Greedy Egyptian fraction decomposition."""
        result = []
        remaining = Fraction(target)
        for _ in range(max_terms):
            if remaining == 0:
                break
            # Find smallest d >= min_denom such that 1/d <= remaining
            d = max(min_denom, 1 / remaining) if remaining > 0 else min_denom
            d = int(math.ceil(1 / remaining)) if remaining > 0 else min_denom
            d = max(d, min_denom)
            if d in result:
                d += 1
            if d > 10000000:
                break
            result.append(d)
            remaining -= Fraction(1, d)
            min_denom = d + 1
        if remaining == 0 and len(result) <= 256:
            return result
        return None
    
    # Try greedy
    greedy_result = greedy_egyptian(1, 5)
    if greedy_result is not None:
        return greedy_result
    
    # Last resort: return a minimal valid set
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Let's just return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # and hope the sum is close enough... no, it must be exactly 1.
    
    # Raise if we can't find a valid solution
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END
