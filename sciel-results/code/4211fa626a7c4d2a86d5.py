import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    # Precompute smallest prime factor (SPF) for fast factorization
    # Max B = n*x. x < 3n/4, so B < 3n^2/4. For n=256, B < 49152.
    # Let's be safe and use a larger limit, say 200000, to handle any edge cases or if x goes higher.
    MAX_B = 200000
    spf = list(range(MAX_B + 1))
    for i in range(2, int(math.isqrt(MAX_B)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Fast factorization using SPF table."""
        if num <= 1:
            return []
        factors = []
        temp = num
        while temp > 1:
            if temp > MAX_B:
                # Fallback for numbers larger than our SPF table (should not happen often if bounds are correct)
                # Simple trial division
                p = 2
                while p * p <= temp:
                    if temp % p == 0:
                        count = 0
                        while temp % p == 0:
                            temp //= p
                            count += 1
                        factors.append((p, count))
                if temp > 1:
                    factors.append((temp, 1))
                return factors
            
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
        return factors
    
    def get_divisors_from_factorization(factors):
        """Generate all divisors from prime factorization."""
        divisors = [1]
        for p, exp in factors:
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors
    
    # Cache for factorizations of B values
    factor_cache = {}
    
    def get_factorization(B):
        if B in factor_cache:
            return factor_cache[B]
        factors = factorize(B)
        factor_cache[B] = factors
        return factors
    
    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # x must satisfy: n/4 < x < 3n/4
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        # Try x values
        for x in range(x_start, x_end + 1):
            # Time check in inner loop
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # Get factorization of B
            b_factors = get_factorization(B)
            b_sq_factors = [(p, 2 * e) for p, e in b_factors]
            
            # Get divisors of B^2
            divisors = get_divisors_from_factorization(b_sq_factors)
            
            # Sort divisors to allow early break
            divisors.sort()
            
            # For each divisor u of B^2 with u <= B:
            # v = B^2 // u
            # y = (u + B) / A must be integer
            # z = (v + B) / A must be integer
            # Check x < y < z
            
            for u in divisors:
                if u > B:
                    break
                v = (B * B) // u
                
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Check distinctness and ordering: x < y < z
                if x < y < z:
                    # Verify the equation holds (sanity check)
                    # 1/x + 1/y + 1/z should equal 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
