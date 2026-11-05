import time

def construct():
    start_time = time.time()
    deadline = start_time + 45
    
    best_max = float('inf')
    best_sol = None
    
    # 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c
    # c = 31ab / (5ab - 31a - 31b)
    # For a: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # b(5a - 31) > 31a for D > 0
        # b < 62a/(5a-31) for c > b
        denom = 5 * a - 31
        if denom <= 0:
            continue
        b_min = max(a + 1, 31 * a // denom + 1)
        b_max = 62 * a // denom
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            D = 5 * a * b - 31 * a - 31 * b
            if D <= 0:
                continue
            num = 31 * a * b
            if num % D == 0:
                c = num // D
                if c > b and c <= 10000000:
                    max_den = c
                    if max_den < best_max:
                        best_max = max_den
                        best_sol = [a, b, c]
    
    # If no 3-term solution found or best is still high, try 4-term
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # For efficiency, only try if best_max > some threshold
    if best_max > 100:
        for a in range(7, 20):
            if time.time() > deadline:
                break
            ra = 5 * a - 31
            if ra <= 0:
                continue
            for b in range(a + 1, a + 100):
                if time.time() > deadline:
                    break
                rab = 5 * a * b - 31 * a - 31 * b
                if rab <= 0:
                    continue
                # 1/c + 1/d = (5ab - 31a - 31b)/(31ab) = rab/(31ab)
                # d = 31ab*c / (rab*c - 31ab)
                # For c: c(5ab - 31a - 31b) - 31ab > 0 => c > 31ab/rab
                # And c < b*2*31ab/rab for d > c... let's bound c
                c_min = max(b + 1, 31 * a * b // rab + 1)
                c_max = min(2 * 31 * a * b // rab, b + 500)
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        break
                    D = rab * c - 31 * a * b
                    if D <= 0:
                        continue
                    num = 31 * a * b * c
                    if num % D == 0:
                        d = num // D
                        if d > c and d <= 10000000:
                            max_den = d
                            if max_den < best_max:
                                best_max = max_den
                                best_sol = [a, b, c, d]
    
    if best_sol is None:
        return [8, 31, 248]
    return best_sol
