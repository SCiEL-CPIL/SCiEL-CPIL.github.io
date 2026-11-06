"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    answers = []
    LIMIT = 10**12
    
    for n in range(3, 257):
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            # Note: if 4x - n <= 0, no solution with y,z positive?
            # If 4x - n <= 0, then 1/x >= 4/n, so 1/y + 1/z <= 0, impossible for positive y,z.
            # So we need 4x > n, which is guaranteed by x > n/4.
            
            A = 4 * x - n
            B = n * x
            
            # 1/y + 1/z = A/B
            # y < z => 2/y > A/B => y < 2B/A
            # 1/y < A/B => y > B/A
            # So y is in (B/A, 2B/A)
            
            min_y = B // A + 1
            max_y = (2 * B) // A - 1
            
            # Iterate y in the valid range
            # Optimization: If range is too large, it might be slow, but for n<=256 it's fine.
            for y in range(min_y, max_y + 1):
                # 1/z = A/B - 1/y = (Ay - B) / (By)
                # z = By / (Ay - B)
                denom = A * y - B
                if denom <= 0:
                    continue
                num = B * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= LIMIT:
                        # Check distinctness: x < y < z is already enforced by loop structure
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
