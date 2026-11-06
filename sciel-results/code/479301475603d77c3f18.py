import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute primes up to 1000 for factoring? 
    # b <= 256 * 192 = 49152. sqrt(49152) ~ 222.
    # Simple trial division is fine.
    
    def get_divisors_of_square(b):
        # Returns list of divisors of b^2
        # First factor b
        n = b
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        # Now generate divisors of b^2
        # If b = p1^e1 * p2^e2 * ..., then b^2 = p1^(2e1) * p2^(2e2) * ...
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            current_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Get divisors of b^2
            # Optimization: b^2 divisors can be many? 
            # Max b ~ 50000. b^2 ~ 2.5e9.
            # Max divisors of 2.5e9 is around 1000-2000.
            # This is fast.
            divs = get_divisors_of_square(b)
            
            for k in divs:
                # Conditions on k:
                # 1. (k + b) % a == 0
                # 2. (b^2 // k + b) % a == 0
                # 3. y = (k+b)//a > x
                # 4. z = (b^2//k + b)//a > y
                # 5. z <= 10^12
                
                # Sign logic for z > y:
                # z - y = (b^2 - k^2) / (a k) > 0
                # If k > 0: b^2 - k^2 > 0 => k < b
                # If k < 0: b^2 - k^2 < 0 => k^2 > b^2 => k < -b
                
                if k > 0:
                    if k >= b:
                        continue
                else:
                    if k >= -b:
                        continue
                        
                if (k + b) % a != 0:
                    continue
                
                y = (k + b) // a
                if y <= x:
                    continue
                
                # Check z
                # b^2 is divisible by k
                term = b * b // k
                if (term + b) % a != 0:
                    continue
                
                z = (term + b) // a
                
                if z <= y:
                    continue
                
                if z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
