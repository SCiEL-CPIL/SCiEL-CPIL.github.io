def construct():
    import time
    from fractions import Fraction
    from math import gcd

    target = Fraction(5, 31)
    p, q = target.numerator, target.denominator  # 5, 31
    start_time = time.time()
    deadline = start_time + 55.0

    best_max = float('inf')
    best_denoms = []

    def try_update(denoms):
        nonlocal best_max, best_denoms
        if len(denoms) < 1 or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        if any(d < 1 or d > 10000000 for d in denoms):
            return
        # Verify sum
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_denoms = denoms[:]

    # --- Phase 1: 3-term enumeration ---
    # 1/d1 + 1/d2 + 1/d3 = 5/31
    # For fixed d1, d2: 1/d3 = 5/31 - 1/d1 - 1/d2
    # d3 = 1 / (5/31 - 1/d1 - 1/d2)
    # = 1 / ((5*d1*d2 - 31*d2 - 31*d1) / (31*d1*d2))
    # = 31*d1*d2 / (5*d1*d2 - 31*d2 - 31*d1)
    # Need denominator > 0 and divides numerator evenly, and d3 distinct from d1,d2

    # d1 must satisfy 1/d1 < 5/31 => d1 >= 7
    # Also for 3 terms, d1 can't be too large or d2,d3 become huge
    # Let's bound: if d1 >= 200, then 5/31 - 1/d1 < 5/31 - 1/200 ≈ 0.158
    # Then 1/d2 + 1/d3 = that, so d2 >= ceil(1/0.158) ≈ 7
    # But d3 = 31*d1*d2 / (5*d1*d2 - 31*d2 - 31*d1)
    # For large d1,d2, numerator ~ 31*d1*d2, denominator ~ 5*d1*d2, so d3 ~ 31/5 = 6.2
    # So d3 would be around 6-7, but must be > d1 (if we assume ordering d1<d2<d3)
    # Actually we don't need ordering, just distinctness.
    # Let's be generous: d1 from 7 to 500, d2 from d1+1 to 50000

    for d1 in range(7, 501):
        if time.time() > deadline:
            break
        # d2 must be > d1 (to avoid duplicates) and 1/d2 < 5/31 - 1/d1
        # 1/d2 < (5*d1 - 31)/(31*d1) => d2 > 31*d1/(5*d1 - 31)
        # Also 1/d2 + 1/d3 = 5/31 - 1/d1, and d3 > d2
        # So 2/d2 > 5/31 - 1/d1 => d2 < 2/(5/31 - 1/d1) = 62*d1/(5*d1 - 31)
        num1 = 5 * d1 - 31
        if num1 <= 0:
            continue
        d2_min = (31 * d1) // num1 + 1  # d2 > 31*d1/num1
        d2_max = (62 * d1) // num1  # d2 < 62*d1/num1, so d2 <= this
        
        if d2_min < d1 + 1:
            d2_min = d1 + 1
        if d2_max < d2_min:
            continue
        
        # Cap d2_max to avoid too many iterations
        d2_max = min(d2_max, 200000)
        
        for d2 in range(d2_min, d2_max + 1):
            if time.time() > deadline:
                break
            # Compute d3 = 31*d1*d2 / (5*d1*d2 - 31*d2 - 31*d1)
            denom = 5 * d1 * d2 - 31 * d2 - 31 * d1
            if denom <= 0:
                continue
            numer = 31 * d1 * d2
            if numer % denom != 0:
                continue
            d3 = numer // denom
            if d3 <= d2:
                continue
            if d3 > 10000000:
                continue
            denoms = [d1, d2, d3]
            try_update(denoms)

    # --- Phase 2: 4-term enumeration ---
    # 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31
    # For fixed d1, d2, d3: 1/d4 = 5/31 - 1/d1 - 1/d2 - 1/d3
    # d4 = 1 / (5/31 - 1/d1 - 1/d2 - 1/d3)
    # Use Fraction for precision
    for d1 in range(7, 101):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        # 1/d2 + 1/d3 + 1/d4 = r1
        # d2 > d1, and 1/d2 < r1 => d2 > 1/r1
        d2_min = (r1.denominator + r1.numerator - 1) // r1.numerator
        if d2_min < d1 + 1:
            d2_min = d1 + 1
        # 2/d2 > r1 - 1/d3 (for d3 > d2) => roughly d2 < 2/r1
        d2_max = min((2 * r1.denominator) // r1.numerator + 1, 5000)
        
        for d2 in range(d2_min, d2_max + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            # 1/d3 + 1/d4 = r2
            # d3 > d2, 1/d3 < r2 => d3 > 1/r2
            d3_min = (r2.denominator + r2.numerator - 1) // r2.numerator
            if d3_min < d2 + 1:
                d3_min = d2 + 1
            # 2/d3 > r2 => d3 < 2/r2
            d3_max = min((2 * r2.denominator) // r2.numerator + 1, 10000)
            
            for d3 in range(d3_min, d3_max + 1):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, d3)
                if r3 <= 0:
                    continue
                # 1/d4 = r3, so d4 = 1/r3
                if r3.numerator != 1:
                    continue
                d4 = r3.denominator
                if d4 <= d3:
                    continue
                if d4 > 10000000:
                    continue
                denoms = [d1, d2, d3, d4]
                try_update(denoms)

    # --- Phase 3: Greedy with lookahead for 5+ terms ---
    if time.time() < deadline - 5:
        for d1 in range(7, 50):
            if time.time() > deadline - 2:
                break
            r = target - Fraction(1, d1)
            if r <= 0:
                continue
            used = {d1}
            denoms = [d1]
            
            while r > 0:
                k_min = (r.denominator + r.numerator - 1) // r.numerator
                best_k = None
                best_max_after = float('inf')
                
                # Try a few candidates
                for candidate_k in range(k_min, min(k_min + 15, 10000001)):
                    if candidate_k in used:
                        continue
                    temp_r = r - Fraction(1, candidate_k)
                    if temp_r < 0:
                        continue
                    if temp_r == 0:
                        if candidate_k < best_max_after:
                            best_max_after = candidate_k
                            best_k = candidate_k
                        continue
                    # Greedy complete rest, check max
                    temp_used = used | {candidate_k}
                    temp_max = candidate_k
                    tr = temp_r
                    valid = True
                    for _ in range(30):
                        if tr <= 0:
                            break
                        k2 = (tr.denominator + tr.numerator - 1) // tr.numerator
                        while k2 in temp_used:
                            k2 += 1
                        if k2 > 10000000:
                            valid = False
                            break
                        temp_used.add(k2)
                        if k2 > temp_max:
                            temp_max = k2
                        tr -= Fraction(1, k2)
                    if valid and tr == 0 and temp_max < best_max_after:
                        best_max_after = temp_max
                        best_k = candidate_k
                
                if best_k is None:
                    k = k_min
                    while k in used:
                        k += 1
                    best_k = k
                denoms.append(best_k)
                used.add(best_k)
                r -= Fraction(1, best_k)
            
            if denoms:
                try_update(denoms)

    # Fallback
    if not best_denoms:
        r = target
        best_denoms = []
        used = set()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            best_denoms.append(d)
            used.add(d)
            r -= Fraction(1, d)

    return best_denoms
