import time
import itertools

def construct():
    start_time = time.time()
    timeout = 60.0 

    # 1. Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    smooth_set = set(smooth_numbers)
    
    answers = []
    
    for target in range(1, 129):
        # Time check
        if time.time() - start_time > timeout - 5.0:
            # Fallback: use a simple valid representation
            # For safety, we can use a known valid pattern
            # But this shouldn't happen
            break
            
        found = None
        
        # Try 1 term
        if target in smooth_set:
            found = [target]
        else:
            # Try 2 terms: a + b = target, a < b, both smooth, antichain
            # a must be < target/2
            for a in smooth_numbers:
                if a * 2 >= target:
                    break
                b = target - a
                if b in smooth_set and b > a:
                    # Check antichain: since a < b, only need b % a != 0
                    if b % a != 0:
                        found = [a, b]
                        break
            
            # Try 3 terms: a + b + c = target, a < b < c, all smooth, antichain
            if found is None:
                for a in smooth_numbers:
                    if a * 3 >= target:
                        break
                    rem1 = target - a
                    for b in smooth_numbers:
                        if b <= a:
                            continue
                        if b * 2 >= rem1:
                            break
                        c = rem1 - b
                        if c in smooth_set and c > b:
                            # Check antichain for (a, b, c)
                            if b % a != 0 and c % a != 0 and c % b != 0:
                                found = [a, b, c]
                                break
                    if found is not None:
                        break
            
            # Try 4 terms as fallback
            if found is None:
                # a < b < c < d, a+b+c+d = target
                for a in smooth_numbers:
                    if a * 4 >= target:
                        break
                    rem1 = target - a
                    for b in smooth_numbers:
                        if b <= a:
                            continue
                        if b * 3 >= rem1:
                            break
                        rem2 = rem1 - b
                        for c in smooth_numbers:
                            if c <= b:
                                continue
                            if c * 2 >= rem2:
                                break
                            d = rem2 - c
                            if d in smooth_set and d > c:
                                # Check antichain for (a, b, c, d)
                                # All pairs must not divide each other
                                if (b % a != 0 and c % a != 0 and d % a != 0 and
                                    c % b != 0 and d % b != 0 and d % c != 0):
                                    found = [a, b, c, d]
                                    break
                        if found is not None:
                            break
                    if found is not None:
                        break
        
        answers.append(found)

    return answers
