import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')
    
    # Baseline: pure greedy
    r = target
    vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        vals.append(d)
        r -= Fraction(1, d)
    best_solution = vals[:]
    best_max_den = max(vals)
    
    def try_solution(sol):
        nonlocal best_solution, best_max_den
        if not sol:
            return
        # Verify
        s = sum(Fraction(1, d) for d in sol)
        if s != target:
            return
        if len(set(sol)) != len(sol):
            return
        if any(d < 1 or d > 10000000 for d in sol):
            return
        md = max(sol)
        if md < best_max_den:
            best_max_den = md
            best_solution = sol[:]
    
    # Search for solutions with max denominator < best_max_den
    max_search = min(best_max_den - 1, 1000)
    
    # 2-term
    for a in range(7, min(30, max_search + 1)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        if r1.denominator % r1.numerator == 0:
            b = r1.denominator // r1.numerator
            if b > a and b <= max_search:
                try_solution([a, b])
    
    # 3-term
    for a in range(7, min(30, max_search + 1)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        for b in range(a + 1, min(100, max_search + 1)):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            if r2.denominator % r2.numerator == 0:
                c = r2.denominator // r2.numerator
                if c > b and c <= max_search:
                    try_solution([a, b, c])
    
    # 4-term
    for a in range(7, min(25, max_search + 1)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        for b in range(a + 1, min(80, max_search + 1)):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            p, q = r2.numerator, r2.denominator
            for d in range(b + 1, max_search + 1):
                if time.time() > deadline:
                    break
                num = p * d - q
                if num <= 0:
                    continue
                if (q * d) % num != 0:
                    continue
                c = (q * d) // num
                if c <= b or c >= d:
                    continue
                try_solution([a, b, c, d])
    
    # 5-term: a, b, c, d, e
    for a in range(7, min(20, max_search + 1)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        for b in range(a + 1, min(60, max_search + 1)):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            for c in range(b + 1, min(100, max_search + 1)):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                p, q = r3.numerator, r3.denominator
                for e in range(c + 1, max_search + 1):
                    if time.time() > deadline:
                        break
                    num = p * e - q
                    if num <= 0:
                        continue
                    if (q * e) % num != 0:
                        continue
                    d = (q * e) // num
                    if d <= c or d >= e:
                        continue
                    try_solution([a, b, c, d, e])
    
    if best_solution is None:
        return vals
    
    return best_solution
