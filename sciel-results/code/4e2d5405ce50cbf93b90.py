"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MAX_DENOM = 10**12
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    answers = [None] * 254  # for n=3..256
    
    def factorize(num):
        """Factorize num into prime factors using trial division."""
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 1)
        return factors
    
    def get_divisors(factors):
        """Generate all divisors from prime factorization."""
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            p_power = 1
            for e in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= prime
            divisors = new_divs
        return divisors
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() - start_time > DEADLINE:
            break
        found = None
        # Try multiple values of x starting from the smallest valid
        # x must satisfy 4/x > 4/n, so x < n. Also 1/x + 1/y + 1/z = 4/n
        # with y > x, z > y, so 3/x > 4/n => x < 3n/4
        # Also 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # since 3/x > 4/n means x < 3n/4
        
        # Try x values from x_min upward
        for x in range(x_min, min(x_max + 1, 100000)):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Factorize b_sq
            b_sq_factors = factorize(b_sq)
            divisors = get_divisors(b_sq_factors)
            
            for d in divisors:
                # y = (d + b) / a
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= 0 or y <= x:
                    continue
                if y > MAX_DENOM:
                    continue
                
                # z = (b_sq // d + b) / a
                b_sq_div_d = b_sq // d
                if (b_sq_div_d + b) % a != 0:
                    continue
                z = (b_sq_div_d + b) // a
                if z <= 0 or z <= y:
                    continue
                if z > MAX_DENOM:
                    continue
                
                # Verify distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Verify the equation by cross multiplication
                # 1/x + 1/y + 1/z = 4/n
                # n*(y*z + x*z + x*y) = 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
# EVOLVE-BLOCK-END
