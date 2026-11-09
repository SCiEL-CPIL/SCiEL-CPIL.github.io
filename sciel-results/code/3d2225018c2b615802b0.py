import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    deadline = start_time + 85.0

    def get_pairs_for_remainder(rem, min_b, max_c):
        """
        Find all pairs (b, c) with min_b <= b < c <= max_c such that 1/b + 1/c = rem.
        Returns a list of tuples (b, c).
        """
        if rem <= 0:
            return []
        p = rem.numerator
        q = rem.denominator
        q2 = q * q
        pairs = []
        
        # Iterate divisors of q^2
        # To avoid duplicates, we can just iterate all divisors and filter, 
        # or iterate d <= sqrt(q^2) and get both.
        # Since we need b < c, we can assume the factor assigned to b is smaller?
        # No, b and c are symmetric in the equation, but we want b < c.
        # (p*b - q) * (p*c - q) = q^2.
        # Let X = p*b - q, Y = p*c - q.
        # X * Y = q^2.
        # b = (X + q) / p, c = (Y + q) / p.
        # We want b < c => X < Y.
        # So we iterate X < Y.
        
        limit = isqrt(q2)
        for i in range(1, limit + 1):
            if q2 % i != 0:
                continue
            X = i
            Y = q2 // i
            if X >= Y:
                continue
            
            # Check integrality
            if (X + q) % p != 0:
                continue
            if (Y + q) % p != 0:
                continue
            
            b = (X + q) // p
            c = (Y + q) // p
            
            if b <= 0 or c <= 0:
                continue
            if b == c:
                continue
            
            if b < min_b or c > max_c:
                continue
            if b >= c: # Should be handled by X < Y, but just in case
                continue
                
            pairs.append((b, c))
            
        return pairs

    # Search for minimal largest denominator M
    # We check M from 1 upwards.
    # For a given M, we check if there exists a solution with max <= M.
    
    for M in range(1, 2000):
        if time.time() > deadline:
            break
            
        # Check 2 terms
        # 1/a + 1/b = target, a < b <= M
        # Use get_pairs_for_remainder with rem=target, min_b=1, max_c=M
        # But get_pairs_for_remainder expects rem = 1/b+1/c.
        # So for 2 terms, we just call it with min_b=1.
        # Wait, the function finds b,c such that 1/b+1/c=rem.
        # So for 2 terms, rem=target.
        pairs_2 = get_pairs_for_remainder(target, 1, M)
        if pairs_2:
            # Any pair (b,c) from this is a valid 2-term solution with max <= M
            # We return the one with smallest max? 
            # The problem asks to minimize the LARGEST denominator.
            # Since we are iterating M, the first M that yields a solution is the minimal M.
            # Within that M, does it matter which solution?
            # "Return 1..256 distinct denominators... Minimize the LARGEST denominator".
            # The score is based on the value (max denominator).
            # So any solution with max <= M is fine, as long as we return it.
            # The evaluator measures the raw value (max denominator).
            # So we can just return the first one found.
            b, c = pairs_2[0]
            return [b, c]

        # Check 3 terms
        # 1/a + 1/b + 1/c = target, a < b < c <= M
        # Iterate a from 1 to M
        # For each a, rem2 = target - 1/a
        # Find pairs (b,c) with a < b < c <= M such that 1/b + 1/c = rem2
        for a in range(1, M + 1):
            rem2 = target - Fraction(1, a)
            if rem2 <= 0:
                break
            
            # We need b > a and c <= M
            # get_pairs_for_remainder(rem2, a + 1, M)
            pairs_3 = get_pairs_for_remainder(rem2, a + 1, M)
            if pairs_3:
                b, c = pairs_3[0]
                # Verify distinctness: a < b < c is enforced by min_b=a+1 and b<c in generator
                # Verify sum: exact by construction
                return [a, b, c]
                
        # Check 4 terms?
        # 1/a + 1/b + 1/c + 1/d = target
        # If M is small, 4 terms might be needed?
        # 5/31 ~ 0.16.
        # 1/7 ~ 0.14.
        # 1/8 ~ 0.125.
        # 1/9 ~ 0.11.
        # 1/10 ~ 0.1.
        # Sum of 4 small fractions.
        # If we didn't find 2 or 3 terms, we might need 4.
        # But 3 terms usually suffice for these sizes.
        # Let's add a 4-term check just in case, but it's more expensive.
        
        # For 4 terms: a < b < c < d <= M
        # Iterate a
        # Iterate b
        # rem3 = target - 1/a - 1/b
        # Find pairs (c,d) with b < c < d <= M
        found_4 = False
        for a in range(1, M + 1):
            if time.time() > deadline:
                break
            rem2 = target - Fraction(1, a)
            if rem2 <= 0:
                break
            # b must be > a
            # Also 1/b < rem2 => b > 1/rem2
            # And 3/b >= rem2 (since c,d > b) => b <= 3/rem2
            b_max = min(M, int(3 / rem2.numerator * rem2.denominator) + 1)
            for b in range(a + 1, b_max + 1):
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                pairs_4 = get_pairs_for_remainder(rem3, b + 1, M)
                if pairs_4:
                    c, d = pairs_4[0]
                    return [a, b, c, d]
                    
    # Fallback
    return [8, 31, 248]
