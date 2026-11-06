import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')
    
    def update_best(denoms):
        nonlocal best_solution, best_max_den
        m = max(denoms)
        if m < best_max_den:
            best_max_den = m
            best_solution = denoms[:]
            
    # Search for 3-term solutions
    # 1/a + 1/b + 1/c = target
    # a < b < c
    # a >= 7 (since 1/6 > 5/31)
    # 3/a > target => a < 3*31/5 = 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() > deadline: break
        r1 = target - Fraction(1, a)
        if r1 <= 0: continue
        # 1/b + 1/c = r1
        # b > a
        # 1/b < r1 => b > 1/r1
        # 2/b > r1 => b < 2/r1
        b_min = int(1 / r1) + 1
        if b_min <= a: b_min = a + 1
        b_max = int(2 / r1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline: break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0: continue
            # 1/c = r2
            if r2.numerator == 1:
                c = r2.denominator
                if c > b:
                    update_best([a, b, c])

    # Search for 4-term solutions
    # 1/a + 1/b + 1/c + 1/d = target
    for a in range(7, 19):
        if time.time() > deadline: break
        r1 = target - Fraction(1, a)
        if r1 <= 0: continue
        # 1/b + 1/c + 1/d = r1
        # b > a
        # 1/b < r1 => b > 1/r1
        # 3/b > r1 => b < 3/r1
        b_min = int(1 / r1) + 1
        if b_min <= a: b_min = a + 1
        b_max = int(3 / r1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline: break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0: continue
            # 1/c + 1/d = r2
            # c > b
            # 1/c < r2 => c > 1/r2
            # 2/c > r2 => c < 2/r2
            c_min = int(1 / r2) + 1
            if c_min <= b: c_min = b + 1
            c_max = int(2 / r2)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline: break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0: continue
                if r3.numerator == 1:
                    d = r3.denominator
                    if d > c:
                        update_best([a, b, c, d])

    # Search for 5-term solutions
    # 1/a + 1/b + 1/c + 1/d + 1/e = target
    for a in range(7, 19):
        if time.time() > deadline: break
        r1 = target - Fraction(1, a)
        if r1 <= 0: continue
        # 1/b + 1/c + 1/d + 1/e = r1
        b_min = int(1 / r1) + 1
        if b_min <= a: b_min = a + 1
        b_max = int(4 / r1) # 4/b > r1
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline: break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0: continue
            # 1/c + 1/d + 1/e = r2
            c_min = int(1 / r2) + 1
            if c_min <= b: c_min = b + 1
            c_max = int(3 / r2)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline: break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0: continue
                # 1/d + 1/e = r3
                d_min = int(1 / r3) + 1
                if d_min <= c: d_min = c + 1
                d_max = int(2 / r3)
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline: break
                    r4 = r3 - Fraction(1, d)
                    if r4 <= 0: continue
                    if r4.numerator == 1:
                        e = r4.denominator
                        if e > d:
                            update_best([a, b, c, d, e])

    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if no solution found in time (should not happen)
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
