import time

def construct():
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    for idx in range(254):
        n = idx + 3
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct, WLOG ordered)
        # This implies 1/x < 4/n => x > n/4
        # Also 1/x + 1/y + 1/z < 3/x => 3/x > 4/n => x < 3n/4
        # And since x<y<z, 1/x > 1/3 * 4/n => x < 3n/4 is a loose bound.
        # Actually 1/x > 1/y > 1/z, so 1/x > (4/n)/3 = 4/(3n) => x < 3n/4.
        # Lower bound: x > n/4.
        
        start_x = n // 4 + 1
        # Upper bound for x: since y > x and z > y, 1/y < 1/x and 1/z < 1/x
        # So 4/n = 1/x + 1/y + 1/z < 3/x => x < 3n/4
        # But also 4/n > 1/x + 1/(x+1) + 1/(x+2) roughly? No, that's for existence.
        # A tighter upper bound: 1/x < 4/n - 1/(x+1) - 1/(x+2) is not helpful directly.
        # Let's just iterate x from start_x upwards.
        # For each x, we need 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx. We need 1/y + 1/z = A/B.
        # This requires A > 0, so x > n/4.
        # (y+z)/(yz) = A/B => Ayz - By - Bz = 0 => Ayz - By - Bz + B^2/A = B^2/A (if A divides B^2? No)
        # Multiply by A: A^2 yz - AB y - AB z = 0
        # (Ay - B)(Az - B) = B^2
        # Let u = Ay - B, v = Az - B. Then uv = B^2.
        # y = (u+B)/A, z = (v+B)/A.
        # We need u, v to be divisors of B^2 such that:
        # 1. (u+B) % A == 0 and (v+B) % A == 0
        # 2. y = (u+B)/A is an integer > x (since y > x)
        # 3. z = (v+B)/A is an integer > y
        # 4. y, z <= 10^12
        
        # B = nx. B^2 can be very large. n up to 256, x up to ~192. B up to ~49152. B^2 up to ~2.4e9.
        # Number of divisors of B^2 is manageable.
        
        for x in range(start_x, int(3*n/4) + 2):
            if time.time() > deadline:
                break
            A = 4*x - n
            if A <= 0:
                continue
            B = n*x
            
            # We need to find divisors u of B^2 such that (u+B) % A == 0.
            # And v = B^2 // u, such that (v+B) % A == 0.
            # And y = (u+B)//A > x, z = (v+B)//A > y.
            
            # To ensure y < z, we need u < v, so u < B.
            # Iterate through divisors u of B^2 with u <= B.
            
            # Optimized divisor iteration:
            # B is at most 256 * (3*256/4) = 256 * 192 = 49152.
            # B^2 is at most ~2.4e9.
            # Iterating up to sqrt(B^2) = B is fine.
            
            # Actually, we can just iterate u from 1 to B.
            # But checking B^2 % u == 0 for all u up to B is O(B) which is ~50k.
            # 254 * 50 * 50k = 635 million operations. Might be too slow in Python.
            
            # Better: factorize B, generate divisors.
            # Or just iterate u up to B and check divisibility.
            # Let's try the simple loop first, it might be fast enough for small x.
            # Most solutions will have small x.
            
            # Alternative: iterate divisors of B^2 properly.
            # Let's factorize B.
            # B = n * x. n and x are small.
            
            # Let's use a faster method:
            # We need u | B^2 and (u+B) % A == 0.
            # Let u = A*k - B. Then A*k - B | B^2.
            # This is getting complicated.
            
            # Let's stick to iterating u up to B, but break early if found.
            # Also, we can skip u if (u+B) % A != 0.
            
            # Optimization: u must be such that u ≡ -B (mod A).
            # So u = A*k - B for some k.
            # Also u | B^2.
            # And 1 <= u <= B.
            
            # Let's iterate k such that u = A*k - B is in [1, B].
            # 1 <= A*k - B <= B
            # B+1 <= A*k <= 2B
            # (B+1)/A <= k <= 2B/A
            
            k_min = (B + 1) // A + 1 if (B+1) % A != 0 else (B+1)//A
            # Actually: k > B/A => k >= floor(B/A) + 1
            # k <= 2B/A => k <= floor(2B/A)
            
            k_start = B // A + 1
            k_end = (2*B) // A
            
            for k in range(k_start, k_end + 1):
                u = A * k - B
                if u <= 0 or u > B:
                    continue
                if (B*B) % u != 0:
                    continue
                v = (B*B) // u
                # Check v condition: (v+B) % A == 0
                # Note: v = B^2/u. Since u | B^2, v is integer.
                # Also u*v = B^2.
                # (v+B) % A == 0 ?
                # We know u ≡ -B (mod A).
                # v = B^2/u. This doesn't immediately give v mod A.
                # Just check it.
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness and order
                if y > x and z > y and y != z:
                    # Check bounds
                    if y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers[idx] = found
    
    return answers
