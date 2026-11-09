"""Optimized search for 1/x+1/y+1/z=4/n using linear scan in y."""
import time

def construct():
    answers = []
    deadline = time.time() + 55  # safety margin
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # and 1/x + 1/y + 1/z = 4/n with y,z > 0 implies 3/x > 4/n => x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        if end_x < start_x:
            end_x = start_x
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            P = 4 * x - n
            if P <= 0:
                continue
            Q = n * x
            # We need y such that z = Q*y / (P*y - Q) is a positive integer
            # P*y - Q > 0 => y > Q/P
            y_min = Q // P + 1
            # Cap y to avoid infinite loops; z must be >= 1, so P*y - Q <= Q*y
            # => P*y <= Q*y + Q => y*(Q-P) >= Q => y >= Q/(Q-P) if Q > P
            # Also z = Q*y/(P*y - Q) <= 10^12
            # Just cap y at a reasonable bound
            y_max = min(10**7, Q // P + 100000)
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = P * y - Q
                if denom <= 0:
                    continue
                num = Q * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= 0:
                    continue
                if y == z:
                    continue
                if y == x or z == x:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        answers.append(found)
    return answers
