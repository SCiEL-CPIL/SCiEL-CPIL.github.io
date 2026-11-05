"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    # Precompute primes up to a limit sufficient for factorizing B = n*x
    # In our strategy, we iterate x from n+1 to some bound.
    # Max n = 256. If we pick x ~ n, B ~ n^2 ~ 65536.
    # To be safe, let's sieve up to 100,000.
    limit = 100000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def factorize(num):
        if num <= 1:
            return []
        factors = []
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        return factors

    def get_divisors_from_factorization(factors):
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

    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # We want 1/x + 1/y + 1/z = 4/n with x < y < z
        # Rearranging for z:
        # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # (y+z)/(yz) = A/B
        # B(y+z) = Ayz
        # Ayz - By - Bz = 0
        # Multiply by A:
        # A^2 yz - AB y - AB z = 0
        # Add B^2:
        # (Ay - B)(Az - B) = B^2
        
        # Let u = Ay - B and v = Az - B. Then uv = B^2.
        # y = (u+B)/A, z = (v+B)/A.
        # We need y, z to be integers, so u+B and v+B must be divisible by A.
        # Also x < y < z implies:
        # 1. x < y => x < (u+B)/A => Ax < u+B => u > Ax - B = A(x) - nx = x(4-n) (since A=4x-n, B=nx, Ax-B = x(4x-n) - nx = 3x^2 - 2nx? No.)
        # Let's stick to the simpler check: calculate y, z and verify x<y<z.
        
        # Iterate x. 
        # Lower bound for x: 1/x < 4/n => x > n/4. So x >= n//4 + 1.
        # Upper bound for x: Since x<y<z, 1/x > 1/3 * (4/n) => x < 3n/4.
        # So x is in (n/4, 3n/4).
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4  # Exclusive upper bound effectively, or just iterate up to n
        
        # Actually, if x >= n, 1/x <= 1/n, then 1/y+1/z >= 3/n. Since y>x>=n, 1/y+1/z < 2/n. Contradiction.
        # So x < n is a hard limit.
        
        for x in range(x_start, n):
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # Factor B^2. Note that factors of B^2 are squares of factors of B? No.
            # If B = prod p_i^e_i, B^2 = prod p_i^(2e_i).
            # We can factorize B and double exponents.
            b_factors = factorize(B)
            b_sq_factors = [(p, 2*e) for p, e in b_factors]
            
            divisors = get_divisors_from_factorization(b_sq_factors)
            
            # We need u * v = B^2.
            # We can iterate u up to sqrt(B^2) = B.
            # For each divisor u of B^2 such that u <= B:
            # v = B^2 // u
            # y = (u + B) / A
            # z = (v + B) / A
            # Check if y, z are integers.
            # Check if x < y < z.
            
            # Optimization: 
            # u + B must be divisible by A.
            # u ≡ -B (mod A)
            # u ≡ -nx (mod 4x-n)
            
            # Instead of checking all divisors, we can just iterate divisors of B^2.
            # Number of divisors of B^2 is manageable for B ~ 65536.
            # Max divisors for n<100k is around 1000-2000.
            
            for u in divisors:
                if u > B:
                    break # Since we iterate sorted divisors? get_divisors doesn't guarantee sort.
                
            # Let's sort divisors to allow breaking early if u > B (since v = B^2/u < B, so u must be <= B for u<=v)
            divisors.sort()
            
            for u in divisors:
                if u > B:
                    break
                v = (B * B) // u
                
                # y = (u + B) / A
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                # z = (v + B) / A
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Check distinctness and order
                if x < y < z:
                    found = [x, y, z]
                    break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
# EVOLVE-BLOCK-END
