import time
import math

def construct():
    answers = [None] * 254
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s wall-clock

    for idx, n in enumerate(range(3, 257)):
        if time.time() - t0 > DEADLINE:
            break
        found = None
        x_min = n // 4 + 1
        
        # Search over x values
        x_search_limit = x_min + 5000
        for x in range(x_min, x_search_limit):
            if time.time() - t0 > DEADLINE:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Reduce fraction a/b
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            # 1/y + 1/z = a_red / b_red
            # Solutions: for each divisor d of a_red * b_red^2:
            #   y = (b_red + d) / a_red  (must be integer)
            #   z = b_red * (b_red + d) / (a_red * d)  (must be integer)
            
            # a_red * b_red^2 can be large, so we factor and generate divisors
            # But that's expensive. Instead, let's iterate d over divisors of a_red*b_red^2
            # by iterating over divisors of a_red and b_red^2 separately.
            
            # Alternative: iterate y in a smart way
            # y must satisfy: a_red * y - b_red > 0 and (a_red * y - b_red) | b_red * y
            # Let d = a_red * y - b_red, so y = (d + b_red) / a_red
            # d must divide b_red * (d + b_red) / a_red
            # => a_red * d | b_red * (d + b_red)
            # => a_red * d | b_red * d + b_red^2
            # => a_red * d | b_red^2 (since a_red*d | b_red*d trivially)
            
            # So d must be a divisor of b_red^2 / gcd(a_red, b_red^2)
            # Actually: a_red * d | b_red^2 means d | b_red^2 / gcd(a_red, b_red^2)
            
            # Let's compute this
            g2 = math.gcd(a_red, b_red * b_red)
            if g2 == 0:
                continue
            # d must divide (b_red^2) // g2 ... wait let me re-derive
            
            # We need: a_red * d divides b_red * y = b_red * (d + b_red) / a_red
            # For y to be integer: a_red | (d + b_red), i.e., d ≡ -b_red (mod a_red)
            # For z to be integer: a_red * d | b_red * (d + b_red)
            #   => a_red * d | b_red * d + b_red^2
            #   Since a_red * d | b_red * d iff a_red | b_red (which is true since we reduced),
            #   we need a_red * d | b_red^2
            #   i.e., d | b_red^2 / gcd(a_red, b_red^2)
            
            target = (b_red * b_red) // g2
            
            # Now iterate over divisors d of target such that d ≡ -b_red (mod a_red)
            # and y = (d + b_red) / a_red > x (for distinctness and ordering)
            
            # Factor target to get divisors
            # This could be expensive for large target, so let's cap it
            
            # For small target, enumerate divisors directly
            if target <= 10**12:
                # Find all divisors of target
                divisors = []
                i = 1
                while i * i <= target:
                    if target % i == 0:
                        divisors.append(i)
                        if i != target // i:
                            divisors.append(target // i)
                    i += 1
                
                for d in divisors:
                    if time.time() - t0 > DEADLINE:
                        break
                    # Check d ≡ -b_red (mod a_red)
                    if (d + b_red) % a_red != 0:
                        continue
                    y = (d + b_red) // a_red
                    if y <= x:
                        continue
                    # z = b_red * y / d
                    if (b_red * y) % d != 0:
                        continue
                    z = (b_red * y) // d
                    if z <= y:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    found = [x, y, z]
                    break
            else:
                # Fallback: scan y directly but with a limit
                y_lo = b_red // a_red + 1
                scan_end = y_lo + min(200000, 10**7)
                for y in range(y_lo, scan_end):
                    if time.time() - t0 > DEADLINE:
                        break
                    denom = a_red * y - b_red
                    if denom <= 0:
                        continue
                    num = b_red * y
                    if num % denom != 0:
                        continue
                    z = num // denom
                    if z <= y:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found

    return answers
