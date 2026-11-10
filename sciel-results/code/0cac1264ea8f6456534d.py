import time
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 55
    
    target = Fraction(5, 31)
    
    best_solution = [8, 28, 1736]
    best_max = 1736
    
    # Search for 3-term solutions
    # 1/d1 + 1/d2 + 1/d3 = 5/31
    # d1 < d2 < d3
    
    # d1 bounds
    # 3/d1 > 5/31 => d1 < 18.6
    # 1/d1 < 5/31 => d1 > 6.2
    
    for d1 in range(7, 19):
        if time.time() - start_time > timeout:
            break
        
        R1 = target - Fraction(1, d1)
        if R1 <= 0:
            continue
        
        # 1/d2 + 1/d3 = R1
        # (R1.den * d2 - R1.num)(R1.den * d3 - R1.num) = R1.den^2
        # Let A = R1.num, B = R1.den
        # (B*d2 - A)(B*d3 - A) = B^2
        A = R1.numerator
        B = R1.denominator
        product = B * B
        
        # Find divisors k2 of product such that:
        # 1. k2 < sqrt(product)
        # 2. k2 + A is divisible by B
        # 3. d2 = (k2 + A) / B > d1
        # 4. d3 = (product//k2 + A) / B > d2
        
        # Iterate through divisors
        # Since product is at most (31*18)^2 ~ 3000^2 = 9e6, iteration is fast
        # But we can just iterate k2 up to sqrt(product)
        
        import math
        limit = int(math.isqrt(product))
        
        for k2 in range(1, limit + 1):
            if time.time() - start_time > timeout:
                break
            if product % k2 != 0:
                continue
            
            k3 = product // k2
            
            # Check divisibility for d2
            if (k2 + A) % B != 0:
                continue
            d2 = (k2 + A) // B
            
            # Check divisibility for d3
            if (k3 + A) % B != 0:
                continue
            d3 = (k3 + A) // B
            
            if d2 <= d1 or d3 <= d2:
                continue
            
            # Verify distinctness and sum
            if d1 != d2 and d2 != d3 and d1 != d3:
                if Fraction(1, d1) + Fraction(1, d2) + Fraction(1, d3) == target:
                    current_max = d3
                    if current_max < best_max:
                        best_max = current_max
                        best_solution = [d1, d2, d3]
    
    return best_solution
