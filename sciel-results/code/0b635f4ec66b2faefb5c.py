import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    # Precompute smallest prime factor for numbers up to max B
    # Max n=256, max x approx 3*256/4 = 192. Max B = 256*192 = 49152.
    # Let's use 60000 to be safe.
    MAX_B = 60000
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_divisors_of_square(B):
        if B == 1:
            return [1]
        factors = {}
        temp = B
        while temp > 1:
            p = spf[temp]
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            factors[p] = cnt
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            power = 1
            for _ in range(exp + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # We need Y * Z = B2
            # y = (Y+B)/A, z = (Z+B)/A
            # Conditions:
            # 1. y, z integers => Y % A == (-B) % A, Z % A == (-B) % A
            # 2. y > x, z > x => Y > 4x^2 - 2nx, Z > 4x^2 - 2nx
            # 3. y != z => Y != Z
            # 4. y, z <= LIMIT => Y <= A*LIMIT - B, Z <= A*LIMIT - B
            
            # Get divisors of B2
            divs = get_divisors_of_square(B)
            
            # Threshold for Y > 4x^2 - 2nx
            thresh = 4 * x * x - 2 * n * x
            max_Y = A * LIMIT - B
            
            r = (-B) % A
            
            for Y in divs:
                if Y <= thresh:
                    continue
                if Y > max_Y:
                    continue
                if Y % A != r:
                    continue
                
                Z = B2 // Y
                if Z == Y:
                    continue
                if Z <= thresh:
                    continue
                if Z > max_Y:
                    continue
                if Z % A != r:
                    continue
                
                # Calculate y, z
                y = (Y + B) // A
                z = (Z + B) // A
                
                # Verify distinctness and bounds
                if y == z:
                    continue
                if y > x and z > x and y <= LIMIT and z <= LIMIT:
                    # Ensure x, y, z are distinct
                    # We know y > x, z > x.
                    # We checked y != z.
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[n - 3] = found

    return answers
