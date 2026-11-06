"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin before 60s kill (90s hard kill)

    target = Fraction(5, 31)

    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_den = 1736

    def get_divisors(n):
        """Get all divisors of n sorted."""
        if n <= 0:
            return []
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def complete_two_terms(remainder, used_set, max_denom):
        """
        Solve 1/x + 1/y = remainder exactly via divisor enumeration of q^2.
        Returns list of valid (x, y) pairs with x < y, distinct, not in used,
        both <= max_denom, both > 0.
        """
        if remainder <= 0:
            return []
        p = remainder.numerator
        q = remainder.denominator
        q2 = q * q
        divs = get_divisors(q2)
        pairs = []
        for A in divs:
            B = q2 // A
            if A > B:
                continue  # avoid duplicates
            num_x = A + q
            if num_x % p != 0:
                continue
            x = num_x // p
            num_y = B + q
            if num_y % p != 0:
                continue
            y = num_y // p
            if x <= 0 or y <= 0:
                continue
            if x > max_denom or y > max_denom:
                continue
            if x in used_set or y in used_set:
                continue
            if x == y:
                continue
            if x > y:
                x, y = y, x
            pairs.append((x, y))
        return pairs

    def update_candidate(denoms):
        """Update best solution if candidate is valid and better."""
        nonlocal best_solution, best_max_den
        if not denoms or len(denoms) > 256:
            return
        if any(d <= 0 or d > 10000000 for d in denoms):
            return
        if len(set(denoms)) != len(denoms):
            return
        s = sum(Fraction(1, d) for d in denoms)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max_den:
            best_max_den = mx
            best_solution = denoms[:]

    # --- Phase 1: Exhaustive 3-term search ---
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    # a >= 7 (since 1/6 > 5/31)
    # For each a, 1/b + 1/c = rem_ab
    # b >= a+1, and 2/b > rem_ab => b < 2/rem_ab
    a_max = 200
    for a in range(7, a_max + 1):
        if time.time() - start_time > time_limit:
            break
        rem_ab = target - Fraction(1, a)
        if rem_ab <= 0:
            continue
        # b must satisfy 1/b < rem_ab => b > 1/rem_ab, and b > a
        b_min = a + 1
        # 2/b > rem_ab => b < 2/rem_ab
        b_max = min(2 // rem_ab.numerator * rem_ab.denominator, 5000)
        b_max = max(b_max, b_min)
        for b in range(b_min, b_max + 1):
            if time.time() - start_time > time_limit:
                break
            rem_bc = rem_ab - Fraction(1, b)
            if rem_bc <= 0:
                continue
            used = {a, b}
            pairs = complete_two_terms(rem_bc, used, best_max_den - 1)
            for (x, y) in pairs:
                update_candidate([a, b, x, y])

    # --- Phase 2: Exhaustive 4-term search ---
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # For each (a, b), complete 1/c + 1/d = rem via algebraic 2-term
    a_max4 = 80
    for a in range(7, a_max4 + 1):
        if time.time() - start_time > time_limit:
            break
        rem_ab = target - Fraction(1, a)
        if rem_ab <= 0:
            continue
        b_min = a + 1
        b_max4 = min(2 // rem_ab.numerator * rem_ab.denominator, 500)
        b_max4 = max(b_max4, b_min)
        for b in range(b_min, b_max4 + 1):
            if time.time() - start_time > time_limit:
                break
            rem_bc = rem_ab - Fraction(1, b)
            if rem_bc <= 0:
                continue
            # c must satisfy 1/c < rem_bc => c > 1/rem_bc, and c > b
            c_min = b + 1
            c_max = min(2 // rem_bc.numerator * rem_bc.denominator, 2000)
            c_max = max(c_max, c_min)
            for c in range(c_min, c_max + 1):
                if time.time() - start_time > time_limit:
                    break
                rem_cd = rem_bc - Fraction(1, c)
                if rem_cd <= 0:
                    continue
                used = {a, b, c}
                pairs = complete_two_terms(rem_cd, used, best_max_den - 1)
                for (x, y) in pairs:
                    update_candidate([a, b, c, x, y])

    # --- Phase 3: Greedy completion for broader coverage ---
    def greedy_complete(remainder, used_set, max_denom):
        """Complete a remainder using greedy, respecting max_denom constraint."""
        if remainder == 0:
            return []
        if remainder < 0:
            return None
        values = []
        r = remainder
        while r > 0:
            d_min = (r.denominator + r.numerator - 1) // r.numerator
            if d_min < 1:
                d_min = 1
            found = False
            for d in range(d_min, min(d_min + 200, max_denom + 1)):
                if d in used_set:
                    continue
                new_r = r - Fraction(1, d)
                if new_r < 0:
                    continue
                values.append(d)
                used_set.add(d)
                r = new_r
                found = True
                break
            if not found:
                return None
            if len(values) > 200:
                return None
        return values

    # Try single starting denominators with greedy completion
    for d1 in range(7, 100):
        if time.time() - start_time > time_limit:
            break
        rem = target - Fraction(1, d1)
        if rem <= 0:
            continue
        used = {d1}
        completion = greedy_complete(rem, used.copy(), 10000000)
        if completion is not None:
            all_denoms = [d1] + completion
            if len(set(all_denoms)) == len(all_denoms) and len(all_denoms) <= 256:
                update_candidate(all_denoms)

    # Try pairs of starting denominators with greedy completion
    for d1 in range(7, 60):
        if time.time() - start_time > time_limit:
            break
        rem1 = target - Fraction(1, d1)
        if rem1 <= 0:
            continue
        d2_min = d1 + 1
        d2_max = min(200, int(rem1.denominator / rem1.numerator) + 50)
        d2_max = max(d2_max, d2_min)
        for d2 in range(d2_min, d2_max + 1):
            if time.time() - start_time > time_limit:
                break
            rem2 = rem1 - Fraction(1, d2)
            if rem2 < 0:
                continue
            if rem2 == 0:
                update_candidate([d1, d2])
                continue
            used = {d1, d2}
            completion = greedy_complete(rem2, used.copy(), 10000000)
            if completion is not None:
                all_denoms = [d1, d2] + completion
                if len(set(all_denoms)) == len(all_denoms) and len(all_denoms) <= 256:
                    update_candidate(all_denoms)

    # Return the best solution found
    return best_solution
# EVOLVE-BLOCK-END
