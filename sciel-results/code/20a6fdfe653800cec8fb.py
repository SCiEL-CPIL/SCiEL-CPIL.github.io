import math
import time

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        
        # Try multiple x values
        # x must satisfy: 4/n - 1/x > 0, so x > n/4
        # Also 1/x + 1/y + 1/z = 4/n with y,z > x (wlog x<=y<=z)
        # So 3/x >= 4/n => x <= 3n/4
        # And 1/x < 4/n => x > n/4
        start_x = n // 4 + 1
        end_x = min(3 * n // 4 + 1, start_x + 200)
        
        for x in range(start_x, end_x):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # => (y+z)/(yz) = A/B
            # => B(y+z) = A*yz
            # => A*yz - B*y - B*z = 0
            # => (Ay - B)(Az - B) = B²
            # Let U = Ay - B, V = Az - B, so U*V = B²
            # y = (U+B)/A, z = (V+B)/A
            # Need U+B ≡ 0 (mod A) and V+B ≡ 0 (mod A)
            # And y, z positive integers, distinct from each other and from x
            
            B_sq = B * B
            
            # Factor B to enumerate divisors of B²
            # Get prime factorization of B
            temp = B
            factors = {}
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    temp //= d
                d += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Divisors of B²: for each prime p^e in B, p^(2e) in B²
            # Generate all divisors
            divisors = [1]
            for p, e in factors.items():
                exp = 2 * e
                current = [1]
                for i in range(1, exp + 1):
                    current.append(current[-1] * p)
                new_divs = []
                for d1 in divisors:
                    for d2 in current:
                        new_divs.append(d1 * d2)
                divisors = new_divs
            
            # Sort divisors for early termination if needed
            # Check each divisor U of B²
            for U in divisors:
                if U * U > B_sq:
                    break  # U > V, already checked
                if B_sq % U != 0:
                    continue
                V = B_sq // U
                
                # Check y = (U + B) / A
                if (U + B) % A == 0:
                    y = (U + B) // A
                    if y > 0:
                        # Check z = (V + B) / A
                        if (V + B) % A == 0:
                            z = (V + B) // A
                            if z > 0:
                                if y != z and y != x and z != x:
                                    if y <= 10**12 and z <= 10**12:
                                        found = [x, y, z]
                                        break
                if found:
                    break
            if found:
                break
        
        answers.append(found)
    return answers
