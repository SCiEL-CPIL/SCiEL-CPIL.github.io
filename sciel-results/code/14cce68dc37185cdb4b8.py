import math
import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # Search for x in a reasonable range
        # x must be > n/4
        start_x = n // 4 + 1
        # Try a small range of x values
        # We can afford to try more x's now if it's faster, but 50 was enough.
        # Let's stick to 50 to be safe, or maybe 100?
        # With divisor search, 100 is very fast.
        for x in range(start_x, start_x + 100):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            if A == 1:
                y = B + 1
                z = B * (B + 1)
                if y > 0 and z > 0 and y != z and y != x and z != x:
                    if y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
                continue

            # Factor B to get divisors of B^2
            # B is small (< 30000)
            # Simple trial division for factoring
            b_fact = {}
            temp_b = B
            d = 2
            while d * d <= temp_b:
                while temp_b % d == 0:
                    b_fact[d] = b_fact.get(d, 0) + 1
                    temp_b //= d
                d += 1
            if temp_b > 1:
                b_fact[temp_b] = b_fact.get(temp_b, 0) + 1
            
            # Generate divisors of B^2
            # If B = prod p_i^e_i, then B^2 = prod p_i^(2e_i)
            divs = [1]
            for p, e in b_fact.items():
                exp = 2 * e
                new_divs = []
                p_power = 1
                for k in range(exp + 1):
                    for d0 in divs:
                        new_divs.append(d0 * p_power)
                    p_power *= p
                divs = new_divs
            
            # Check divisors
            B_sq = B * B
            found_x = False
            for u in divs:
                if (u + B) % A != 0:
                    continue
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if y > 0 and z > 0:
                    if y != z and y != x and z != x:
                        if y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            found_x = True
                            break
            if found:
                break
        answers.append(found)
    return answers
