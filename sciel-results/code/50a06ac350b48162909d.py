import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin
    
    answers = [None] * 254  # for n=3..256
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        # Try multiple candidate values for x (smallest denominator)
        # x must satisfy 1/x < 4/n, so x > n/4
        # Also x should be reasonably small for efficiency
        
        # Try x from floor(n/4)+1 up to some limit
        x_min = n // 4 + 1
        # Try a range of x values
        max_x = min(n // 4 + 20, 200)  # limit to keep search fast
        
        for x in range(x_min, max_x + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n  # must be positive
            if A <= 0:
                continue
            
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # So Ay - B must be a positive divisor of B^2
            # And y = (d + B) / A must be integer and > x
            # And z = (B^2/d + B) / A must be integer and > y
            
            B_sq = B * B
            
            # Find divisors of B^2 that give valid y, z
            # To be efficient, iterate over divisors d of B^2
            # But B^2 can be large, so we iterate over divisors of B and use them
            
            # Actually, let's iterate over possible d values more carefully
            # d must divide B^2, and d = Ay - B, so d ≡ -B (mod A)
            # Also d > 0 (since y > x implies Ay > Ax, and we need Ay - B > 0)
            
            # Let's find all divisors of B^2
            # But that's expensive. Instead, iterate over divisors of B
            # and consider d = divisor, d = B^2/divisor
            
            # Better approach: iterate over divisors d of B^2
            # For efficiency, get divisors of B first, then form divisors of B^2
            
            # Let's just iterate over a limited range of d values
            # d = Ay - B, so d ≡ -B (mod A), i.e., d ≡ A - (B mod A) (mod A) if B mod A != 0
            # Actually d mod A = (-B) mod A = (A - B%A) % A
            
            rem = (-B) % A  # d must satisfy d % A == rem
            
            # d must be a positive divisor of B^2 with d % A == rem
            # And y = (d + B) / A > x, z = (B^2/d + B) / A > y
            
            # Iterate over divisors of B^2
            # Get divisors of B first
            divisors_B = []
            i = 1
            while i * i <= B:
                if B % i == 0:
                    divisors_B.append(i)
                    if i != B // i:
                        divisors_B.append(B // i)
                i += 1
            
            # Divisors of B^2 include products of divisors of B
            # But that's complex. Let's just check d values directly
            # d must divide B^2, so d = p^2 * q where p, q divide B with appropriate constraints
            
            # Simpler: iterate over possible d in a smart way
            # d ranges from 1 to B^2, but we only need d % A == rem
            # Start from the smallest d >= 1 with d % A == rem
            # and check if d divides B^2
            
            # The smallest positive d with d % A == rem:
            if rem == 0:
                d_start = A
            else:
                d_start = rem
            
            # Iterate d = d_start, d_start + A, d_start + 2*A, ...
            # Check if d divides B^2
            # Limit the number of iterations
            
            max_iterations = 1000
            count = 0
            d = d_start
            while d <= B_sq and count < max_iterations:
                if B_sq % d == 0:
                    y = (d + B) // A
                    if y * A == d + B:  # exact division
                        if y > x:
                            z_num = B_sq // d + B
                            z = z_num // A
                            if z * A == z_num:  # exact division
                                if z > y:
                                    found = [x, y, z]
                                    break
                d += A
                count += 1
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
