"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Returns the list of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # target = 5/31
    target_num = 5
    target_den = 31
    
    best_max_den = 248  # current known solution [8, 31, 248]
    best_solution = [8, 31, 248]
    
    def try_two_terms(rem_num, rem_den, prev_den, max_allowed, used_set):
        """Try to express rem_num/rem_den as 1/b + 1/c with b > prev_den, b,c <= max_allowed, distinct from used_set."""
        # 1/b + 1/c = rem_num/rem_den
        # => rem_den*(b+c) = rem_num*b*c
        # => b*(rem_num*c - rem_den) = rem_den*c
        # => b = rem_den*c / (rem_num*c - rem_den)
        # For b to be positive integer: rem_num*c - rem_den > 0 => c > rem_den/rem_num
        # Also b > prev_den and b < c (WLOG)
        # b < c => rem_den*c/(rem_num*c - rem_den) < c
        # => rem_den < rem_num*c - rem_den
        # => 2*rem_den < rem_num*c
        # => c > 2*rem_den/rem_num
        
        c_min = max(prev_den + 1, rem_den // rem_num + 1, 2 * rem_den // rem_num + 1)
        # Also need rem_num*c - rem_den > 0
        c_min = max(c_min, rem_den // rem_num + 1)
        
        # b = rem_den*c / (rem_num*c - rem_den)
        # b must be integer, so (rem_num*c - rem_den) | rem_den*c
        # Let g = gcd(rem_num*c - rem_den, rem_den)
        # Then (rem_num*c - rem_den)/g | c
        
        for c in range(c_min, max_allowed + 1):
            if c in used_set:
                continue
            denom_b = rem_num * c - rem_den
            if denom_b <= 0:
                continue
            if (rem_den * c) % denom_b != 0:
                continue
            b = (rem_den * c) // denom_b
            if b <= prev_den:
                continue
            if b >= c:
                continue
            if b > max_allowed:
                continue
            if b in used_set:
                continue
            # Found: b, c
            return [b, c]
        return None
    
    def try_three_terms(max_allowed):
        """Find 1/a + 1/b + 1/c = 5/31 with a<b<c<=max_allowed, minimizing max(c)."""
        # a must satisfy: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
        # Also 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
        # And a < b < c <= max_allowed, so a <= max_allowed - 2
        a_min = 7
        a_max = min(18, max_allowed - 2)
        
        for a in range(a_min, a_max + 1):
            # remaining = 5/31 - 1/a = (5a - 31)/(31a)
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            # Reduce fraction
            g = gcd(rem_num, rem_den)
            rem_num //= g
            rem_den //= g
            
            # Now find 1/b + 1/c = rem_num/rem_den with a < b < c <= max_allowed
            result = try_two_terms(rem_num, rem_den, a, max_allowed, {a})
            if result is not None:
                b, c = result
                return [a, b, c]
        return None
    
    def try_four_terms(max_allowed):
        """Find 1/a + 1/b + 1/c + 1/d = 5/31 with a<b<c<d<=max_allowed."""
        # a: 1/a < 5/31 => a >= 7; 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
        a_min = 7
        a_max = min(24, max_allowed - 3)
        
        for a in range(a_min, a_max + 1):
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            g = gcd(rem_num, rem_den)
            rem_num //= g
            rem_den //= g
            
            # Now find 3 terms summing to rem_num/rem_den with a < b < c < d <= max_allowed
            # b: 1/b < rem_num/rem_den => b > rem_den/rem_num
            # 3/b > rem_num/rem_den => b < 3*rem_den/rem_num
            b_min = rem_den // rem_num + 1
            b_max = min(3 * rem_den // rem_num, max_allowed - 2)
            if b_min < a + 1:
                b_min = a + 1
            
            for b in range(b_min, b_max + 1):
                rem2_num = rem_num * b - rem_den
                rem2_den = rem_den * b
                if rem2_num <= 0:
                    continue
                g2 = gcd(rem2_num, rem2_den)
                rem2_num //= g2
                rem2_den //= g2
                
                # Find 1/c + 1/d = rem2_num/rem2_den with b < c < d <= max_allowed
                result = try_two_terms(rem2_num, rem2_den, b, max_allowed, {a, b})
                if result is not None:
                    c, d = result
                    return [a, b, c, d]
        return None
    
    # Systematically try to find solutions with max_den < 248
    # Try 3 terms first (most likely to have small max denominator)
    for max_d in range(247, 50, -1):
        if time.time() > deadline:
            break
        sol = try_three_terms(max_d)
        if sol is not None:
            best_max_den = max_d
            best_solution = sol
            break
    
    # Try 4 terms
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        sol = try_four_terms(max_d)
        if sol is not None:
            best_max_den = max_d
            best_solution = sol
            break
    
    # Try 5 terms with a similar approach
    def try_five_terms(max_allowed):
        """Find 1/a+1/b+1/c+1/d+1/e = 5/31 with a<b<c<d<e<=max_allowed."""
        a_min = 7
        a_max = min(5, max_allowed - 4)  # 5/a > 5/31 => a < 31, but tighter bound
        # 5/a > 5/31 => a < 31, so a_max = min(30, max_allowed-4)
        a_max = min(30, max_allowed - 4)
        
        for a in range(a_min, a_max + 1):
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            g = gcd(rem_num, rem_den)
            rem_num //= g
            rem_den //= g
            
            b_min = rem_den // rem_num + 1
            b_max = min(4 * rem_den // rem_num, max_allowed - 3)
            if b_min < a + 1:
                b_min = a + 1
            
            for b in range(b_min, b_max + 1):
                rem2_num = rem_num * b - rem_den
                rem2_den = rem_den * b
                if rem2_num <= 0:
                    continue
                g2 = gcd(rem2_num, rem2_den)
                rem2_num //= g2
                rem2_den //= g2
                
                c_min = rem2_den // rem2_num + 1
                c_max = min(3 * rem2_den // rem2_num, max_allowed - 2)
                if c_min < b + 1:
                    c_min = b + 1
                
                for c in range(c_min, c_max + 1):
                    rem3_num = rem2_num * c - rem2_den
                    rem3_den = rem2_den * c
                    if rem3_num <= 0:
                        continue
                    g3 = gcd(rem3_num, rem3_den)
                    rem3_num //= g3
                    rem3_den //= g3
                    
                    result = try_two_terms(rem3_num, rem3_den, c, max_allowed, {a, b, c})
                    if result is not None:
                        d, e = result
                        return [a, b, c, d, e]
        return None
    
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        sol = try_five_terms(max_d)
        if sol is not None:
            best_max_den = max_d
            best_solution = sol
            break
    
    return best_solution
# EVOLVE-BLOCK-END
