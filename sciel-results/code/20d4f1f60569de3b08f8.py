import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # Try to find a solution with k=5
    # 1/a + 1/b + 1/c + 1/d + 1/e = 1
    # 5 <= a < b < c < d < e <= 10^7
    
    for a in range(5, 100):
        if time.time() > deadline:
            break
        rem1 = 1.0 - 1.0/a
        if rem1 <= 0: continue
        
        for b in range(a+1, 100):
            if time.time() > deadline:
                break
            rem2 = rem1 - 1.0/b
            if rem2 <= 0: continue
            
            for c in range(b+1, 100):
                if time.time() > deadline:
                    break
                rem3 = rem2 - 1.0/c
                if rem3 <= 0: continue
                
                for d in range(c+1, 1000):
                    if time.time() > deadline:
                        break
                    rem4 = rem3 - 1.0/d
                    if rem4 <= 0: continue
                    
                    # 1/e = rem4
                    # e = 1 / rem4
                    # Check if e is integer
                    # Use fractions for precision
                    from fractions import Fraction
                    target = Fraction(1,1)
                    f_a = Fraction(1,a)
                    f_b = Fraction(1,b)
                    f_c = Fraction(1,c)
                    f_d = Fraction(1,d)
                    f_e = target - f_a - f_b - f_c - f_d
                    
                    if f_e.numerator != 1:
                        continue
                    e = f_e.denominator
                    
                    if e <= d:
                        continue
                    if e > 10000000:
                        continue
                        
                    # Found a solution
                    return [a, b, c, d, e]
                    
    # Fallback: use the DP method from seed but maybe with a better target?
    # Or just return a known valid solution if search fails
    # A known solution for 1 with distinct denoms >= 5:
    # Let's construct one using the DP result or a hardcoded one if found.
    # If no k=5 found, try k=6?
    
    # Fallback to seed logic to ensure validity
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no solution found")
