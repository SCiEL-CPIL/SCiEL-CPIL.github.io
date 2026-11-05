import math
import time

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 4x - n > 0, i.e., x > n/4
        start_x = n // 4 + 1
        
        for x in range(start_x, start_x + 200):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # We need divisors d of B2 such that:
            # y = (d + B) / A is a positive integer
            # z = (B2//d + B) / A is a positive integer
            # y, z > 0, y != z, y != x, z != x, y <= 10^12, z <= 10^12
            
            # Factor B2 to get all divisors
            # B2 = (n*x)^2
            # Let's factor n*x first, then square the prime factors
            bx = B
            # Factor bx
            temp = bx
            factors = {}
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    temp //= d
                d += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Now B2 has prime factorization with doubled exponents
            # Generate all divisors of B2
            # But B2 can be large. Let's be smarter.
            # We only need divisors d where d + B is divisible by A
            # and B2//d + B is divisible by A
            # Also y = (d+B)/A must be > x (to avoid duplicates, assume y > x)
            # and z > y for distinctness ordering
            
            # Actually, let's just iterate over divisors of B2
            # Number of divisors of B2: if B2 = p1^e1 * p2^e2 * ... then (e1+1)(e2+1)...
            # For B up to ~30000, B2 up to ~9*10^8, number of divisors is manageable
            
            # Generate divisors
            prime_powers = []
            for p, e in factors.items():
                powers = [1]
                pe = 1
                for _ in range(2 * e):  # B2 has exponent 2*e
                    pe *= p
                    powers.append(pe)
                prime_powers.append(powers)
            
            # Generate all divisors
            divisors = [1]
            for pp in prime_powers:
                new_divs = []
                for d_old in divisors:
                    for p_pow in pp:
                        new_divs.append(d_old * p_pow)
                divisors = new_divs
            
            # Sort divisors
            divisors.sort()
            
            for d in divisors:
                if time.time() > deadline:
                    break
                # y = (d + B) / A
                if (d + B) % A != 0:
                    continue
                y = (d + B) // A
                if y <= 0:
                    continue
                if y > 10**12:
                    continue
                e_val = B2 // d
                if (e_val + B) % A != 0:
                    continue
                z = (e_val + B) // A
                if z <= 0:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if y == z:
                    continue
                if y == x or z == x:
                    continue
                # Valid solution
                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*y*z + n*x*z + n*x*y == 4*x*y*z
                lhs = n * y * z + n * x * z + n * x * y
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
