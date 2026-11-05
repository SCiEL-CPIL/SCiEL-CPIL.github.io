import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute smallest prime factor (SPF) for fast factorization
    # B = n * x. n <= 256.
    # If we search x up to, say, 1000, B <= 256000.
    # Let's be generous and go up to 5,000,000.
    LIMIT = 5_000_000
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
        If B = p1^a1 * p2^a2, then B^2 = p1^(2a1) * p2^(2a2).
        Divisors are p1^k1 * p2^k2 where 0 <= k1 <= 2a1, etc.
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
            # Fill remaining with None
            for j in range(idx, 254):
                answers[j] = None
            break
            
        found = None
        x_min = n // 4 + 1
        
        # Search x from x_min upwards.
        # Solutions are generally close to n/4.
        # Let's try up to 500 values of x.
        # For n=256, x_min=65. x up to 565. B up to 144k.
        # Factorization is fast.
        max_x_attempts = 500
        
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
            # B is small enough to factorize quickly with SPF
            if B > LIMIT:
                # Fallback for very large B (unlikely with max_x_attempts=500)
                # Use trial division or skip?
                # For now, if B > LIMIT, we can't use SPF directly.
                # But 256 * (256//4 + 500) = 256 * 564 = 144384 < 5,000,000.
                # So B is always within LIMIT.
                pass
            
            factors = factorize(B)
            divs = generate_divisors_of_square(factors)
            
            # We only need to check divisors d in (d_lower_bound, B)
            # Sorting divs is O(D log D) where D is number of divisors.
            # D is small (< 1000 usually).
            divs.sort()
            
            for d in divs:
                if d <= d_lower_bound:
                    continue
                if d >= B:
                    break
                    
                # Check integrality of y and z
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                # Check y > x (should be guaranteed by d > d_lower_bound, but verify)
                if y <= x:
                    continue
                    
                num_z = (B * B) // d + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Check z > y
                if z <= y:
                    continue
                
                # Check distinctness (guaranteed by x < y < z)
                # Check bound <= 10^12
                if z <= 1000000000000:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        answers[idx] = found
        
    return answers
