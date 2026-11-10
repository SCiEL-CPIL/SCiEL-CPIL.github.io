import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    answers = []
    MAX_Z = 10**12
    n_total = 254
    
    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
        
        # Allocate time for this n
        time_per_n = min(remaining / remaining_n, 0.35)
        n_deadline = time.time() + time_per_n
        
        found = None
        
        # x must satisfy: 1/x < 4/n => x > n/4
        # and 1/x + 1/y + 1/z = 4/n with y,z > x implies 3/x > 4/n => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > n_deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            
            G = math.gcd(A, B)
            A_p = A // G
            B_p = B // G
            
            # Need to find D such that:
            # D | B_p^2
            # (D + B_p) % A_p == 0
            # y = (D + B_p) / A_p > x
            # z = B_p * y / D > y
            # x, y, z all distinct and <= MAX_Z
            
            # D = k * A_p - B_p for some k
            # D > 0 => k > B_p / A_p
            # D | B_p^2
            # y = k, so we need k > x and k * A_p - B_p | B_p^2
            
            # Alternative: iterate k from x+1 upward
            # y = k, then D = k*A_p - B_p must divide B_p^2
            # z = B_p * k / D
            
            # But k can be large. Better to iterate divisors of B_p^2.
            # B_p <= n*x/G. For n<=256, x<=192, B_p <= ~49152
            # B_p^2 <= ~2.4e9. sqrt is ~49152.
            
            B_sq = B_p * B_p
            
            # Find all divisors of B_sq efficiently
            # Only consider divisors D where (D + B_p) % A_p == 0
            # and D + B_p > A_p * x (so y > x)
            # and D < B_p * y (so z > y)
            
            limit = int(math.isqrt(B_sq))
            
            # Collect candidate divisors
            candidates = []
            for i in range(1, limit + 1):
                if B_sq % i == 0:
                    d1 = i
                    d2 = B_sq // i
                    for d in (d1, d2):
                        if (d + B_p) % A_p == 0:
                            y = (d + B_p) // A_p
                            if y > x:
                                if (B_p * y) % d == 0:
                                    z = (B_p * y) // d
                                    if z > y and z <= MAX_Z:
                                        if x != y and y != z and x != z:
                                            candidates.append((x, y, z))
            
            if candidates:
                # Pick the first valid one
                found = candidates[0]
                break
        
        answers.append(found)
        
    return answers
