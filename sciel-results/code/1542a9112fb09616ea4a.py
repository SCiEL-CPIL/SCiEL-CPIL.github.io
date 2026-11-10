import time
import math

def construct():
    # Deadline guard with safety margin
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    MAX_Z = 10**12
    n_total = 254  # n from 3 to 256 inclusive
    
    # Time budget per n: leave some slack for overhead
    # If we've spent too much time, reduce budget for remaining n's
    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
        
        # Allocate time for this n, with diminishing budget for later n's
        # Give more time to early n's (they're easier), less to later ones
        # But ensure we don't starve any single n
        time_per_n = min(remaining / remaining_n, 0.5)
        n_deadline = time.time() + time_per_n
        
        found = None
        
        # x must satisfy: x > n/4 and x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in a smart order: smaller x first (gives smaller y ranges)
        for x in range(x_min, x_max + 1):
            if time.time() > n_deadline:
                break
                
            denom_x = 4 * x - n
            if denom_x <= 0:
                continue
            
            # B = n * x
            B = n * x
            A = denom_x
            
            G = math.gcd(A, B)
            A_prime = A // G
            B_prime = B // G
            
            # We need D | B_prime^2
            # And A_prime | D + B_prime
            # And A_prime | B_prime + B_prime^2 / D
            
            # Iterate divisors of B_prime^2
            # B_prime is at most n*x/G. n<=256, x<=192. B<=49152.
            # B_prime^2 <= 2.4e9.
            
            # To find divisors, we can iterate up to sqrt(B_prime^2) = B_prime
            # This is O(B_prime) per x.
            # Sum of B_prime over x might be large?
            # B_prime = n*x/G. G >= 1.
            # Sum_{x} n*x/G ~ n * n^2 / (2 G) ~ n^3 / G.
            # n=256, n^3 ~ 1.6e7. This is acceptable.
            
            B_sq = B_prime * B_prime
            
            # Find divisors of B_sq
            # Optimization: Only check divisors D such that D+B' is divisible by A'
            # D = k*A' - B'
            # D must be positive divisor of B'^2.
            # So k*A' - B' > 0 => k > B'/A'
            # And k*A' - B' | B'^2.
            
            # Instead of finding all divisors, iterate k?
            # D = k*A' - B'.
            # D <= B'^2 => k*A' - B' <= B'^2 => k <= (B'^2 + B')/A'.
            # Number of k is approx B'^2/A'.
            # If A' is small, this is large.
            # If A' is large, this is small.
            
            # Let's stick to finding divisors of B_sq.
            # But we can prune: D must be congruent to -B' mod A'.
            
            # Efficient divisor generation?
            # Standard trial division up to sqrt(B_sq) = B_prime.
            
            divisors = []
            limit = int(math.isqrt(B_sq))
            for i in range(1, limit + 1):
                if B_sq % i == 0:
                    divisors.append(i)
                    if i != B_sq // i:
                        divisors.append(B_sq // i)
                        
            for D in divisors:
                if D <= 0:
                    continue
                if (D + B_prime) % A_prime != 0:
                    continue
                
                y = (D + B_prime) // A_prime
                if y <= x:
                    continue
                
                # Check z
                # z = (B_prime * y) / D
                if (B_prime * y) % D != 0:
                    continue
                
                z = (B_prime * y) // D
                
                if z <= y:
                    continue
                if z > MAX_Z:
                    continue
                
                # Verify distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Verify equation
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply to avoid float issues
                # n*y*z + n*x*z + n*x*y == 4*x*y*z
                lhs = n * y * z + n * x * z + n * x * y
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
