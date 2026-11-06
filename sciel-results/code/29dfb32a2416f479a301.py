import time
import math

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    
    def factorize(num):
        """Return prime factorization as dict {prime: exponent}"""
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    def divisors_of_square(b):
        """Generate all positive divisors of b^2 using prime factorization of b"""
        if b <= 0:
            return [1]
        factors = factorize(b)
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return divs
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in a smart order: start from minimum, which tends to give solutions
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            
            # Get divisors of b^2
            # b is at most 256 * 192 = 49152, so b^2 is at most ~2.4e9
            # Number of divisors is manageable
            divs = divisors_of_square(b)
            
            for k in divs:
                # We need: y = (k + b) / a and z = (b^2/k + b) / a
                # Conditions:
                # 1. (k + b) % a == 0
                # 2. (b^2 // k + b) % a == 0  
                # 3. y > x (distinct and ordered)
                # 4. z > y (ordered)
                # 5. All positive and <= 10^12
                
                if (k + b) % a != 0:
                    continue
                
                y = (k + b) // a
                if y <= x:
                    continue
                
                # k must divide b^2, which it does by construction
                bq = b * b // k
                if (bq + b) % a != 0:
                    continue
                
                z = (bq + b) // a
                
                if z <= y:
                    continue
                
                if z > 10**12 or y > 10**12:
                    continue
                
                # Verify with cross multiplication
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
