def construct():
    import time
    from fractions import Fraction
    from math import gcd
    from heapq import heappush, heappop
    import sys

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
        # Verify sum using integer arithmetic to avoid Fraction overhead
        # Sum of 1/d_i == 5/31 iff 31 * sum(prod/d_i) == 5 * prod
        # But this is expensive for large n. Use Fraction for verification.
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_denoms = denoms[:]

    # ============================================================
    # Phase 1: Exhaustive 3-term search
    # 1/d1 + 1/d2 + 1/d3 = 5/31
    # d3 = 31*d1*d2 / (5*d1*d2 - 31*d2 - 31*d1)
    # ============================================================
    for d1 in range(7, 300):
        if time.time() > deadline:
            break
        num1 = 5 * d1 - 31
        if num1 <= 0:
            continue
        d2_min = 31 * d1 // num1 + 1
        d2_max = 62 * d1 // num1
        if d2_min < d1 + 1:
            d2_min = d1 + 1
        if d2_max < d2_min:
            continue
        d2_max = min(d2_max, 100000)
        for d2 in range(d2_min, d2_max + 1):
            if time.time() > deadline:
                break
            denom = 5 * d1 * d2 - 31 * d2 - 31 * d1
            if denom <= 0:
                continue
            numer = 31 * d1 * d2
            if numer % denom != 0:
                continue
            d3 = numer // denom
            if d3 <= d2 or d3 > 10000000:
                continue
            denoms = [d1, d2, d3]
            try_update(denoms)

    # ============================================================
    # Phase 2: Exhaustive 4-term search with tighter bounds
    # 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31
    # For fixed d1,d2,d3: d4 = 1/(5/31 - 1/d1 - 1/d2 - 1/d3)
    # ============================================================
    for d1 in range(7, 60):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        d2_min = (r1.denominator + r1.numerator - 1) // r1.numerator
        if d2_min < d1 + 1:
            d2_min = d1 + 1
        d2_max = (2 * r1.denominator) // r1.numerator + 1
        d2_max = min(d2_max, 5000)
        for d2 in range(d2_min, d2_max + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            d3_min = (r2.denominator + r2.numerator - 1) // r2.numerator
            if d3_min < d2 + 1:
                d3_min = d2 + 1
            d3_max = (2 * r2.denominator) // r2.numerator + 1
            d3_max = min(d3_max, 20000)
            for d3 in range(d3_min, d3_max + 1):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, d3)
                if r3 <= 0:
                    continue
                # d4 = 1/r3
                if r3.numerator != 1:
                    continue
                d4 = r3.denominator
                if d4 <= d3 or d4 > 10000000:
                    continue
                denoms = [d1, d2, d3, d4]
                try_update(denoms)

    # ============================================================
    # Phase 3: A* search for 5-term solutions with small max
    # State: (remaining_fraction, min_allowed_denom, count_so_far, current_max)
    # Priority: current_max (we want to minimize the largest denominator)
    # ============================================================
    if time.time() < deadline - 5:
        # Use a different strategy: enumerate d1, d2 from small values,
        # then solve for the remaining 3 terms with bounded search
        for d1 in range(7, 30):
            if time.time() > deadline - 3:
                break
            r1 = target - Fraction(1, d1)
            if r1 <= 0:
                continue
            for d2 in range(d1 + 1, min(d1 + 50, 200)):
                if time.time() > deadline - 3:
                    break
                r2 = r1 - Fraction(1, d2)
                if r2 <= 0:
                    continue
                # Now need 1/d3 + 1/d4 + 1/d5 = r2
                # 3-term subproblem
                # d3 > d2
                d3_min = (r2.denominator + r2.numerator - 1) // r2.numerator
                if d3_min < d2 + 1:
                    d3_min = d2 + 1
                d3_max = (2 * r2.denominator) // r2.numerator + 1
                d3_max = min(d3_max, 5000)
                for d3 in range(d3_min, d3_max + 1):
                    if time.time() > deadline - 3:
                        break
                    r3 = r2 - Fraction(1, d3)
                    if r3 <= 0:
                        continue
                    # Need 1/d4 + 1/d5 = r3
                    # d4 > d3
                    d4_min = (r3.denominator + r3.numerator - 1) // r3.numerator
                    if d4_min < d3 + 1:
                        d4_min = d3 + 1
                    d4_max = (2 * r3.denominator) // r3.numerator + 1
                    d4_max = min(d4_max, 50000)
                    for d4 in range(d4_min, d4_max + 1):
                        if time.time() > deadline - 3:
                            break
                        r4 = r3 - Fraction(1, d4)
                        if r4 <= 0:
                            continue
                        if r4.numerator != 1:
                            continue
                        d5 = r4.denominator
                        if d5 <= d4 or d5 > 10000000:
                            continue
                        denoms = [d1, d2, d3, d4, d5]
                        try_update(denoms)

    # ============================================================
    # Phase 4: Greedy with bounded lookahead for 6+ terms
    # Try to find solutions where all denominators are <= some bound M
    # by using a modified greedy that avoids creating huge denominators
    # ============================================================
    if time.time() < deadline - 2:
        # Try different starting denominators and use a smarter completion
        for d1 in range(7, 20):
            if time.time() > deadline - 2:
                break
            r = target - Fraction(1, d1)
            if r <= 0:
                continue
            used = {d1}
            denoms = [d1]
            
            # Use a bounded greedy: at each step, try denominators up to a limit
            # and pick the one that minimizes the eventual max
            max_bound = 2000
            for _ in range(10):  # up to 10 more terms
                if r == 0:
                    break
                k_min = (r.denominator + r.numerator - 1) // r.numerator
                best_k = None
                best_score = float('inf')
                
                # Try candidates from k_min up to k_min + 30, capped at max_bound
                upper = min(k_min + 30, max_bound)
                for candidate in range(k_min, upper + 1):
                    if candidate in used:
                        continue
                    temp_r = r - Fraction(1, candidate)
                    if temp_r < 0:
                        continue
                    if temp_r == 0:
                        score = candidate
                        if score < best_score:
                            best_score = score
                            best_k = candidate
                        continue
                    # Estimate: how many more terms needed, and what's the max
                    tr = temp_r
                    temp_max = candidate
                    valid = True
                    for _ in range(20):
                        if tr <= 0:
                            break
                        k2 = (tr.denominator + tr.numerator - 1) // tr.numerator
                        while k2 in used:
                            k2 += 1
                        if k2 > max_bound:
                            valid = False
                            break
                        used.add(k2)
                        if k2 > temp_max:
                            temp_max = k2
                        tr -= Fraction(1, k2)
                    # Remove temporarily added denominators
                    # (we added them to used, need to clean up)
                    # Actually, let's not add to used in the estimate
                    # Redo: just estimate without modifying used
                    if valid and tr == 0:
                        score = temp_max
                        if score < best_score:
                            best_score = score
                            best_k = candidate
                
                if best_k is None:
                    # Fallback: use greedy
                    k = k_min
                    while k in used:
                        k += 1
                    best_k = k
                    if best_k > max_bound:
                        break
                
                denoms.append(best_k)
                used.add(best_k)
                r -= Fraction(1, best_k)
            
            if r == 0 and denoms:
                try_update(denoms)

    # Fallback: standard greedy
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
