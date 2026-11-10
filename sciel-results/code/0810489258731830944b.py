import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4 (assuming x is the smallest)
        # So x is in [floor(n/4) + 1, floor(3n/4)]
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            # A must be positive, so 4x > n => x > n/4, which is satisfied by x_start
            A = 4 * x - n
            B = n * x
            
            # 1/y < A/B => y > B/A
            # 2/y > A/B => y < 2B/A
            # So y is in [floor(B/A) + 1, floor(2B/A)]
            y_start = B // A + 1
            y_end = (2 * B) // A
            
            # Optimization: if y_start > y_end, continue
            if y_start > y_end:
                continue
                
            for y in range(y_start, y_end + 1):
                # 1/z = A/B - 1/y = (Ay - B) / (By)
                # z = By / (Ay - B)
                denom = A * y - B
                if denom <= 0:
                    continue
                
                # Check if By is divisible by denom
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    # We need z > y to ensure distinctness and ordering x < y < z
                    if z > y:
                        # Check bound
                        if z <= 1000000000000:
                            found = [x, y, z]
                            break
            if found:
                break
        answers.append(found)
    return answers
