"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 82  # safety margin before the 90s kill
    answers = []

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        # x is the smallest denominator: 1/x < 4/n => x > n/4, and 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        # Try larger x first (smaller a = 4x-n) for faster search
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
            if found:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0 or b <= 0:
                continue
            
            b2 = b * b
            
            # d = a*k - b, need d | b², 0 < d < b
            # k range: b/a < k < 2b/a
            # Also need y = (d+b)/a = k to be integer (it is by construction)
            # and z = (b²/d + b)/a to be integer and > y
            
            k_min = b // a + 1  # smallest k such that d = a*k - b > 0
            k_max = (2 * b - 1) // a  # largest k such that d = a*k - b < b
            
            if k_min > k_max:
                continue
            
            # Also need y > x: y = k, so k > x
            if k_min <= x:
                k_min = x + 1
            if k_min > k_max:
                continue
            
            for k in range(k_min, k_max + 1):
                if time.time() > deadline:
                    break
                if found:
                    break
                
                d = a * k - b
                if d <= 0 or d >= b:
                    continue
                
                # Check if d divides b²
                if b2 % d != 0:
                    continue
                
                d2 = b2 // d
                # y = k (since y = (d+b)/a = (a*k - b + b)/a = k)
                y = k
                if y <= x:
                    continue
                
                # z = (d2 + b) / a
                num_z = d2 + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                if z <= y:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Check denominator bound
                if z > 10**12 or y > 10**12 or x > 10**12:
                    continue
                
                # Verify via cross multiplication
                lhs_num = n * y * z + n * x * z + n * x * y
                rhs_num = 4 * x * y * z
                if lhs_num == rhs_num:
                    found = [x, y, z]
                    break
        
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
