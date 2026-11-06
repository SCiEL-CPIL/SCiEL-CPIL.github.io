"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
import math

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n + 3) // 4  # x < 3n/4
        
        # Try x values from smallest upward
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y range: B/A < y <= 2B/A, y > x
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            if y_lo <= x:
                y_lo = x + 1
            if y_lo > y_hi:
                continue
            
            # Fast path: try limited y-scan
            range_size = y_hi - y_lo + 1
            max_scan = 2000 if range_size <= 10000 else 1000
            
            found_for_x = None
            for y in range(y_lo, min(y_hi + 1, y_lo + max_scan)):
                denom = A * y - B
                if denom <= 0:
                    continue
                num = B * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= DENOM_BOUND:
                        if x != y and y != z and x != z:
                            found_for_x = [x, y, z]
                            break
            
            if found_for_x:
                found = found_for_x
                break
            
            # Slow path: divisor-based search for this x
            # (A*y - B)(A*z - B) = B^2
            # Let u = A*y - B, v = A*z - B, u*v = B^2
            # u ≡ -B (mod A), v ≡ -B (mod A)
            # y = (u+B)/A, z = (v+B)/A
            # Need u < B (so y < 2B/A) and u > 0
            # Also y > x => u > A*x - B = A*x - n*x = x*(A-n) = x*(4x-2n) = 2*x*(2*x-n)
            
            b_sq = B * B
            min_u = max(1, 2 * x * (2 * x - n) + 1)
            
            # u must divide b_sq, u in [min_u, B), u ≡ -B (mod A)
            # u = k*A - B for some k
            # k*A - B >= min_u => k >= ceil((min_u + B)/A)
            # k*A - B < B => k < 2*B/A
            # k*A - B > 0 => k > B/A
            
            k_lo = (min_u + B + A - 1) // A
            k_hi = (2 * B - 1) // A
            if k_hi < k_lo:
                continue
            
            # Limit iterations
            max_k_iter = 5000
            for k in range(k_lo, min(k_hi + 1, k_lo + max_k_iter)):
                u = k * A - B
                if u <= 0 or u >= B:
                    continue
                if b_sq % u != 0:
                    continue
                v = b_sq // u
                if (u + B) % A != 0:
                    continue
                if (v + B) % A != 0:
                    continue
                y = (u + B) // A
                z = (v + B) // A
                if y <= x or z <= y:
                    continue
                if y > DENOM_BOUND or z > DENOM_BOUND:
                    continue
                if x == y or y == z or x == z:
                    continue
                found_for_x = [x, y, z]
                break
            
            if found_for_x:
                found = found_for_x
                break
        
        answers[i] = found
    
    return answers
