import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4 (assuming x <= y <= z)
        # So x is in [floor(n/4)+1, floor(3n/4)]
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            # We need (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, V = Az - B. UV = B^2.
            # y = (U + B) / A, z = (V + B) / A
            # Conditions:
            # 1. U divides B^2
            # 2. (U + B) % A == 0
            # 3. (V + B) % A == 0 where V = B^2 // U
            # 4. y > x and z > y (distinct and ordered)
            # 5. z <= 10**12
            
            # To find divisors of B^2, we can factor B.
            # But B is small, so we can just iterate U from 1 to B and check if B^2 % U == 0?
            # No, B^2 can have divisors larger than B. But if U > B, then V < B.
            # We can just iterate U from 1 to B. If U divides B^2, we check.
            # Also check V = B^2 // U.
            # If U < B, V > B. If U > B, V < B.
            # We only need to check U up to B.
            
            # Optimization: We can iterate U from 1 to B.
            # But B can be up to 256 * 192 = 49152.
            # Iterating 49152 times per x is okay if we break early.
            # But we can do better by only checking divisors.
            
            # Let's use a simple divisor iteration for B^2.
            # Since B is small, we can precompute divisors of B^2?
            # No, B changes with x.
            
            # Let's just iterate U from 1 to B.
            # Check if B*B % U == 0.
            # This is O(B) per x.
            # Total O(n^3) which is ~2 million for n=256.
            # Sum over n is ~10^8. Might be tight.
            
            # Better: Factor B.
            # B = n * x.
            # We can factor n once? No, n changes.
            # Factor B directly.
            
            # Let's try the direct iteration first, it's simpler.
            # If it's too slow, we optimize.
            # Actually, we can restrict U further.
            # U must satisfy U ≡ -B (mod A).
            # So U = k*A - B for some k.
            # Also 1 <= U <= B.
            # So k*A - B >= 1 => k >= (B+1)/A.
            # k*A - B <= B => k*A <= 2B => k <= 2B/A.
            # So k is in a small range!
            # This is much faster.
            
            k_min = (B + 1 + A - 1) // A  # ceil((B+1)/A)
            k_max = (2 * B) // A          # floor(2B/A)
            
            for k in range(k_min, k_max + 1):
                U = k * A - B
                if U <= 0:
                    continue
                if (B * B) % U != 0:
                    continue
                V = (B * B) // U
                # Check V + B divisible by A
                if (V + B) % A != 0:
                    continue
                y = (U + B) // A
                z = (V + B) // A
                # Check distinctness
                if x != y and y != z and x != z:
                    if z <= 10**12:
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
    return answers
