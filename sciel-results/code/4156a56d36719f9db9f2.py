"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # We need to find the minimum number of distinct integer denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1.
    
    # Known result: The minimum number of terms in an Egyptian fraction representation of 1
    # with all denominators >= 5 is 3 terms? No, let's check.
    # 1 = 1/5 + 1/6 + 1/... no.
    # Let's think about this mathematically.
    # 1/5 + 1/5 = 2/5 (not distinct)
    # Can we do it in 2 terms? 1/a + 1/b = 1, a,b >= 5, a != b
    # 1/a + 1/b = 1 => a+b = ab => ab - a - b = 0 => (a-1)(b-1) = 1
    # Only solution is a-1=1, b-1=1 => a=2, b=2. Not >= 5 and not distinct.
    # So minimum is at least 3.
    
    # Can we do it in 3 terms? 1/a + 1/b + 1/c = 1, a < b < c, a >= 5
    # 1/a + 1/b + 1/c = 1
    # Since a >= 5, 1/a <= 1/5
    # 1/b + 1/c = 1 - 1/a >= 1 - 1/5 = 4/5
    # But 1/b < 1/a and 1/c < 1/b, so 1/b + 1/c < 2/a <= 2/5 < 4/5. Contradiction.
    # So minimum is at least 4.
    
    # Can we do it in 4 terms? 1/a + 1/b + 1/c + 1/d = 1, a < b < c < d, a >= 5
    # 1/a + 1/b + 1/c + 1/d = 1
    # a >= 5, so 1/a <= 1/5
    # 1/b + 1/c + 1/d = 1 - 1/a >= 4/5
    # 1/b < 1/a, so b > a >= 5, so b >= 6
    # 1/c < 1/b, so c >= 7
    # 1/d < 1/c, so d >= 8
    # Max possible: 1/5 + 1/6 + 1/7 + 1/8 = 0.2 + 0.1667 + 0.1429 + 0.125 = 0.6346 < 1
    # So minimum is at least 5.
    
    # Can we do it in 5 terms? 1/a + 1/b + 1/c + 1/d + 1/e = 1, a < b < c < d < e, a >= 5
    # a >= 5, b >= 6, c >= 7, d >= 8, e >= 9
    # Max possible: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 = 0.7457 < 1
    # So minimum is at least 6.
    
    # Can we do it in 6 terms? 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 1, a < b < c < d < e < f, a >= 5
    # a >= 5, b >= 6, c >= 7, d >= 8, e >= 9, f >= 10
    # Max possible: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 + 0.1 = 0.8457 < 1
    # So minimum is at least 7.
    
    # Can we do it in 7 terms? 1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g = 1, a < b < c < d < e < f < g, a >= 5
    # a >= 5, b >= 6, c >= 7, d >= 8, e >= 9, f >= 10, g >= 11
    # Max possible: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 + 0.1 + 0.0909 = 0.9366 < 1
    # So minimum is at least 8.
    
    # Can we do it in 8 terms? 1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g + 1/h = 1, a < b < c < d < e < f < g < h, a >= 5
    # a >= 5, b >= 6, c >= 7, d >= 8, e >= 9, f >= 10, g >= 11, h >= 12
    # Max possible: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 + 0.1 + 0.0909 + 0.0833 = 1.0200
    # So it's possible in 8 terms. Let's search for it.
    
    # Search for 8 terms: a < b < c < d < e < f < g < h, a >= 5
    # 1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g + 1/h = 1
    
    # Use backtracking with pruning.
    def find_8_terms(time_limit):
        """Find 8 distinct denominators >= 5 whose reciprocal sum is 1."""
        best = None
        
        def dfs(remaining, min_d, current, depth):
            nonlocal best
            if time.time() > time_limit:
                return
            if remaining == 0:
                if depth == 8:
                    best = list(current)
                return
            if depth == 8:
                return
            if remaining <= 0:
                return
            
            # We need to fill (8 - depth) more terms
            terms_needed = 8 - depth
            
            # The minimum possible sum with terms_needed terms starting from min_d
            # is 1/min_d + 1/(min_d+1) + ... + 1/(min_d + terms_needed - 1)
            # The maximum possible sum is 1/min_d + 1/min_d + ... (but they must be distinct)
            # Actually, the maximum sum with terms_needed distinct terms >= min_d is
            # 1/min_d + 1/(min_d+1) + ... + 1/(min_d + terms_needed - 1)
            
            # Pruning: if the maximum possible sum < remaining, no solution
            max_sum = sum(Fraction(1, min_d + i) for i in range(terms_needed))
            if max_sum < remaining:
                return
            
            # Pruning: if the minimum possible sum > remaining, no solution
            # Minimum possible sum with terms_needed distinct terms >= min_d is
            # 1/(large) + 1/(larger) + ... but this approaches 0
            # Actually, we can't easily compute the minimum, but we can use:
            # If 1/min_d > remaining, then even the largest possible term is too big
            if Fraction(1, min_d) > remaining:
                return
            
            # Upper bound on min_d: we need 1/d <= remaining, so d >= ceil(1/remaining)
            # But also d >= min_d
            
            # Lower bound on d: d >= min_d
            # Upper bound on d: we need (terms_needed - 1) more terms after d, each >= d+1
            # The sum of those (terms_needed - 1) terms is at most
            # 1/(d+1) + 1/(d+2) + ... + 1/(d + terms_needed - 1)
            # So 1/d + 1/(d+1) + ... + 1/(d + terms_needed - 1) >= remaining
            # This gives an upper bound on d.
            
            # Simple upper bound: 1/d + (terms_needed - 1)/d >= remaining (rough)
            # Actually, better: 1/d + 1/(d+1) + ... + 1/(d + terms_needed - 1) >= remaining
            # For large d, this is approximately terms_needed/d
            # So d <= terms_needed / remaining (rough)
            
            max_d = int(terms_needed / remaining) + 10  # small buffer
            # But also d must be at most 10000000
            max_d = min(max_d, 10000000)
            
            for d in range(min_d, max_d + 1):
                if time.time() > time_limit:
                    return
                if Fraction(1, d) > remaining:
                    break
                new_rem = remaining - Fraction(1, d)
                current.append(d)
                dfs(new_rem, d + 1, current, depth + 1)
                current.pop()
                if best is not None:
                    return
        
        dfs(Fraction(1, 1), 5, [], 0)
        return best
    
    result = find_8_terms(deadline)
    if result is not None:
        # Verify
        s = sum(Fraction(1, d) for d in result)
        if s == 1 and len(result) == 8 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == 8:
            return result
    
    # Fallback: use a known valid representation with more terms
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/126 + 1/210 + 1/252 + 1/420 + 1/504 + 1/630 + 1/756 + 1/1008 + 1/1260 + 1/1512 + 1/1680 + 1/2520
    fallback = [5, 6, 7, 10, 14, 35, 42, 70, 105, 126, 210, 252, 420, 504, 630, 756, 1008, 1260, 1512, 1680, 2520]
    s = sum(Fraction(1, d) for d in fallback)
    if s == 1:
        return fallback
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
