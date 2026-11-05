import time
import math

# Global cache for the computed answers
_ANSWERS_CACHE = None

def _compute_answers():
    """
    Computes the distinct three-term unit-fraction representations for n=3..256.
    Uses divisor enumeration on B^2 for each candidate x.
    """
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute smallest prime factor (SPF) for fast factorization
    # B = n * x. n <= 256. x is small (starts near n/4).
    # Max x we might check: let's say 1000 steps from n/4.
    # Max B approx 256 * (256/4 + 1000) = 256 * 1064 = 272384.
    # Let's set LIMIT to 1,000,000 to be safe.
    LIMIT = 1_000_000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Factorize num using precomputed SPF. Returns dict of prime->exponent."""
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors[p] = cnt
        return factors
    
    def generate_divisors_of_square(factors):
        """
        Given factors of B, generate all divisors of B^2.
        """
        divs = [1]
        for p, exp in factors.items():
            max_exp = 2 * exp
            new_divs = []
            for d in divs:
                pk = 1
                for k in range(max_exp + 1):
                    new_divs.append(d * pk)
                    pk *= p
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            # Fill remaining with None if we run out of time
            # (Should not happen for this small N, but safety first)
            break
            
        found = None
        x_min = n // 4 + 1
        
        # Search x from x_min upwards.
        # Solutions are generally close to n/4.
        # We'll try up to 1000 values of x to be thorough, 
        # but usually the first few work.
        max_x_attempts = 1000
        
        for i in range(max_x_attempts):
            x = x_min + i
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need d such that:
            # 1. d | B^2
            # 2. (d + B) % A == 0
            # 3. (B^2 // d + B) % A == 0
            # 4. y = (d+B)//A > x
            # 5. z = (B^2//d + B)//A > y
            
            # Condition 4: d + B > x*A => d > x*A - B = x(4x-n) - nx = 4x^2 - 2nx
            d_lower_bound = 4 * x * x - 2 * n * x
            
            # Condition 5: z > y => B^2/d > d => d^2 < B^2 => d < B
            # So we need d_lower_bound < d < B
            
            if d_lower_bound >= B:
                # No possible d in range
                continue
                
            # Factorize B to get divisors of B^2
            if B > LIMIT:
                # Fallback: skip or use trial division. 
                # With max_x_attempts=1000 and n=256, B ~ 256*1064 = 272k < 1M.
                # So this should not trigger.
                continue
            
            factors = factorize(B)
            divs = generate_divisors_of_square(factors)
            
            # Filter divisors in range (d_lower_bound, B)
            # Sorting is O(D log D). D is small.
            # We can also just iterate and check bounds, but sorting allows early exit if we were binary searching,
            # but here we just need to find ANY valid d.
            # Actually, we can just iterate through unsorted divs and check conditions.
            # But checking modulo is the expensive part? No, it's fast.
            # Let's sort to ensure we check smaller d first? 
            # y = (d+B)/A. Smaller d -> smaller y.
            # We just need one solution.
            
            # Optimization: 
            # d must be a divisor of B^2.
            # (d + B) % A == 0 => d % A == (-B) % A.
            # Let r = (-B) % A.
            # We look for divisors d of B^2 such that d % A == r and d_lower_bound < d < B.
            
            r = (-B) % A
            
            # Iterate through divisors
            # To avoid sorting overhead, we can just check all.
            # Number of divisors of B^2 for B~256k is likely < 1000.
            for d in divs:
                if d <= d_lower_bound or d >= B:
                    continue
                if d % A != r:
                    continue
                
                # Check integrality of z
                # z = (B^2/d + B) / A
                # We know (d+B) is divisible by A.
                # Is (B^2/d + B) divisible by A?
                # B^2/d + B = B(B/d + 1).
                # Since d | B^2, let B^2 = d * k. Then z = (k + B)/A.
                # We need (k + B) % A == 0.
                k = (B * B) // d
                if (k + B) % A != 0:
                    continue
                
                y = (d + B) // A
                z = (k + B) // A
                
                if y <= x or z <= y:
                    continue
                
                if z <= 1000000000000:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        answers[idx] = found
        
    return answers

def construct():
    global _ANSWERS_CACHE
    if _ANSWERS_CACHE is None:
        _ANSWERS_CACHE = _compute_answers()
    return _ANSWERS_CACHE
