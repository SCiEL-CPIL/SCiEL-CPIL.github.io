import time
import math

def construct():
    deadline = time.time() + 55.0
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    # Precompute smallest prime factor for fast factorization
    MAX_B = 256 * 500  # B = n*x, x can be up to ~500 for small n
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Factorize num using precomputed SPF. Returns list of (prime, exponent)."""
        if num <= 1:
            return []
        factors = []
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors.append((p, cnt))
        return factors
    
    def get_divisors(num):
        """Returns sorted list of all divisors of num."""
        factors = factorize(num)
        divs = [1]
        for p, exp in factors:
            new_divs = []
            pk = 1
            for k in range(exp + 1):
                for d in divs:
                    new_divs.append(d * pk)
                pk *= p
            divs = new_divs
        divs.sort()
        return divs
    
    for idx in range(254):
        if time.time() > deadline:
            break
            
        n = idx + 3
        found = None
        
        # x must satisfy: 1/x < 4/n => x > n/4, so x_min = n//4 + 1
        # Also need 4x - n > 0 => x > n/4
        x_min = n // 4 + 1
        
        # For efficiency, try small x values first (they give smaller y,z)
        # Limit x attempts to avoid timeout
        max_x = min(x_min + 200, 1000)  # try up to 200 values of x
        
        for x in range(x_min, max_x + 1):
            if time.time() > deadline - 1.0:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let d = Ay - B, then Az - B = B^2/d
            # So d must divide B^2, and d + B ≡ 0 (mod A)
            # Also y = (d+B)/A > x and z = (B^2/d + B)/A > y
            
            # y > x means (d+B)/A > x => d + B > Ax => d > Ax - B = 4x^2 - 2nx
            d_min = 4 * x * x - 2 * n * x + 1
            if d_min < 1:
                d_min = 1
            
            # z > y means B^2/d + B > d + B => B^2/d > d => d^2 < B^2 => d < B
            # So we need d_min <= d < B and d | B^2 and (d+B) % A == 0
            
            # Get divisors of B^2 that are in range [d_min, B)
            # B^2 can be large, so let's factor B^2 directly
            # B = n*x, so B^2 = n^2 * x^2
            # Factorize B^2
            B2 = B * B
            
            # Get all divisors of B^2
            divs = get_divisors(B2)
            
            for d in divs:
                if d < d_min:
                    continue
                if d >= B:
                    break  # divisors are sorted, so we can stop
                
                if (d + B) % A != 0:
                    continue
                
                y = (d + B) // A
                if y <= x:
                    continue
                
                d_prime = B2 // d
                if (d_prime + B) % A != 0:
                    continue
                
                z = (d_prime + B) // A
                if z <= y:
                    continue
                
                # Check distinctness and positivity
                if x < y < z and x > 0 and y > 0 and z > 0:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
