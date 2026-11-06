import math
import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    BOUND = 10**12

    def get_divisors(n):
        """Generate all divisors of n."""
        divs = [1]
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                new_divs = []
                for d in divs:
                    val = 1
                    for _ in range(exp + 1):
                        new_divs.append(d * val)
                        val *= p
                divs = new_divs
            p += 1
        if temp > 1:
            new_divs = []
            for d in divs:
                new_divs.append(d)
                new_divs.append(d * temp)
            divs = new_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            break
            
        found = None
        # x range: n/4 < x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            # We need U | B^2
            # U < B
            # U >= A*(x+1) - B
            # (U+B) % A == 0
            # (B^2/U + B) % A == 0
            
            L = A * (x + 1) - B
            if L < 1:
                L = 1
                
            # If L >= B, no solution for this x with y > x
            if L >= B:
                continue
                
            B_sq = B * B
            divs = get_divisors(B_sq)
            
            # Sort divisors to potentially break early? 
            # We need U in [L, B-1].
            # Since divs are not sorted, we just iterate.
            for U in divs:
                if U >= B:
                    continue
                if U < L:
                    continue
                    
                # Check divisibility for y
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                
                # Check y > x (should be guaranteed by L, but safe check)
                if y <= x:
                    continue
                    
                V = B_sq // U
                # Check divisibility for z
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                
                # Distinctness x,y,z is ensured by x<y<z
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)

    # Fill remaining with None if timeout
    while len(answers) < 254:
        answers.append(None)
        
    return answers
