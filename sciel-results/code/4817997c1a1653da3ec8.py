import math
import time

def construct():
    deadline = time.time() + 70.0  # Safety margin before 90s limit
    answers = []
    
    for n in range(3, 257):
        found = None
        
        # 1/x + 1/y + 1/z = 4/n
        # Assume x <= y <= z
        # Then 1/x >= 1/3 * (4/n) => x <= 3n/4
        # Also 1/x < 4/n => x > n/4
        # So x is in (n/4, 3n/4]
        
        # For a fixed x:
        # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # B(y+z) = Az yz
        # Az yz - B y - B z = 0
        # (Az - B)y - B z = 0
        # (Az - B)y = B z
        # z (Az - B) = By
        # z = By / (Ay - B)
        
        # For z to be an integer, (Ay - B) must divide By.
        # Let D = Ay - B. Then z = By / D.
        # Since z > y (assuming distinct and sorted), By / D > y => B > D => B > Ay - B => 2B > Ay => y < 2B/A.
        # Also D > 0 => Ay > B => y > B/A.
        # So y is in (B/A, 2B/A).
        
        # The range of y is (B/A, 2B/A).
        # Let y_min = floor(B/A) + 1.
        # Let y_max = ceil(2B/A) - 1 (since y < 2B/A strictly if z>y, but let's check boundaries).
        # Actually, if y = 2B/A, then D = A(2B/A) - B = 2B - B = B. z = By/B = y. But we need distinct, so z != y.
        # If z = y, then 2/y = A/B => y = 2B/A.
        # So we need y != 2B/A.
        # Also y > B/A.
        
        # Let's iterate x from smallest possible to largest.
        # Smallest x = floor(n/4) + 1.
        # Largest x = floor(3n/4).
        
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        # Optimization: Iterate x. For each x, the range of y is roughly B/A = nx / (4x-n).
        # As x approaches n/4, A is small, B/A is large.
        # As x approaches 3n/4, A is large, B/A is smaller.
        # Specifically, at x = n/4 + 1 (approx), A=4, B~n^2/4. Range ~ n^2/16.
        # At x = 3n/4, A=2n, B=3n^2/4. Range ~ 3n/8.
        # So smaller x has larger y-range.
        
        # However, the number of divisors of B^2 might be smaller for some x.
        # The equation (Ay-B)(Az-B) = B^2.
        # Let U = Ay - B, V = Az - B.
        # UV = B^2.
        # U < V since y < z.
        # Also U > 0, V > 0.
        # y = (U+B)/A, z = (V+B)/A.
        # We need y, z to be integers.
        # So U ≡ -B (mod A) and V ≡ -B (mod A).
        # Since UV = B^2, and V = B^2/U.
        # (B^2/U) ≡ -B (mod A) => B^2 ≡ -BU (mod A) => B(B+U) ≡ 0 (mod A).
        # This condition is automatically satisfied if A divides B(B+U)? Not necessarily simple.
        # But checking divisors of B^2 is efficient if we can generate them.
        
        # Alternative: Iterate y in range.
        # Range size is B/A = nx / (4x-n).
        # For n=256, x=65 (min), A=260-256=4, B=256*65=16640. Range = 4160.
        # For n=256, x=192 (max), A=768-256=512, B=256*192=49152. Range = 96.
        # The sum of ranges over all x might be manageable.
        
        # Let's try iterating y for each x.
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                # Fallback: return what we have so far, pad with None
                while len(answers) < 254:
                    answers.append(None)
                return answers
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy B/A < y < 2B/A
            # y_min = B // A + 1
            # y_max = (2 * B) // A - 1  (if 2B is divisible by A, y < 2B/A means y <= 2B/A - 1)
            # If 2B not divisible by A, y <= floor(2B/A)
            
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Check if y_max gives z=y
            # If y_max == 2B/A, then z=y. We need distinct, so exclude y_max if it yields z=y.
            # Actually, just check y in range and verify distinctness.
            
            # To speed up, we can iterate y and check if (Ay-B) divides By.
            # D = Ay - B
            # Check if By % D == 0.
            
            # Optimization: Iterate over divisors of B^2?
            # B can be up to 256 * 192 ~ 50000. B^2 ~ 2.5 * 10^9.
            # Factorizing B^2 for each x might be slower than iterating y if range is small.
            # Range is ~ B/A.
            # For small A (small x), range is large.
            # For large A (large x), range is small.
            
            # Let's try iterating y.
            # But wait, the range can be large for small x.
            # n=256, x=65, range 4160.
            # n=256, x=66, A=8, B=16896. Range 2112.
            # n=256, x=70, A=24, B=17920. Range 746.
            
            # Total iterations for n=256:
            # Sum of B/A for x in [65, 192].
            # This sum is roughly integral of nx/(4x-n) dx.
            # Let u = 4x-n, x = (u+n)/4. dx = du/4.
            # Integral n(u+n)/4 / u * du/4 = n/16 * (1 + n/u) du.
            # Limits: x=65 -> u=4. x=192 -> u=512.
            # ~ n/16 * [u + n ln u] from 4 to 512.
            # ~ 256/16 * [512 - 4 + 256 ln(128)] ~ 16 * [508 + 256*4.8] ~ 16 * [508 + 1228] ~ 16 * 1736 ~ 27000.
            # This is very small! 27,000 iterations per n.
            # 254 n's * 27,000 ~ 7 million iterations.
            # Each iteration is a few arithmetic ops. Should be well under 1 second.
            
            for y in range(y_min, y_max + 1):
                D = A * y - B
                if D <= 0:
                    continue
                if (B * y) % D == 0:
                    z = (B * y) // D
                    if z > y:
                        # Check distinctness: x, y, z distinct
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
        
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers

    return answers
