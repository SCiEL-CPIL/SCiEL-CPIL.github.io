def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd

    target = Fraction(5, 31)
    best_max_den = 10000001
    best_solution = None

    def check_solution(denoms):
        nonlocal best_max_den, best_solution
        if len(denoms) < 1 or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        for d in denoms:
            if d < 1 or d > 10000000:
                return
        s = sum(Fraction(1, d) for d in denoms)
        if s == target:
            m = max(denoms)
            if m < best_max_den:
                best_max_den = m
                best_solution = list(denoms)

    def solve_2term(r):
        """Solve 1/b + 1/c = r where r = n/d reduced. Return list of (b,c) pairs."""
        if r <= 0:
            return []
        n = r.numerator
        d = r.denominator
        d2 = d * d
        limit = isqrt(d2)
        results = []
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                if b > 0 and c > 0:
                    if b > c:
                        b, c = c, b
                    results.append((b, c))
        return results

    def complete_with_min_terms(remainder, used_denoms, max_terms=256):
        """Try to complete the remainder using minimal additional terms.
        Returns list of denominators or None if not possible within limits."""
        if remainder == 0:
            return []
        if remainder < 0:
            return None
        if max_terms <= 0:
            return None
        
        # Try 1 term: 1/d = remainder => d = 1/remainder
        if remainder.numerator == 1:
            d = remainder.denominator
            if d not in used_denoms and d <= 10000000:
                return [d]
        
        # Try 2 terms: 1/b + 1/c = remainder
        pairs = solve_2term(remainder)
        for b, c in pairs:
            if b not in used_denoms and c not in used_denoms and b != c:
                if b <= 10000000 and c <= 10000000:
                    return [b, c]
        
        # Try 3 terms via greedy
        r = remainder
        vals = []
        while r > 0 and len(vals) < max_terms:
            d_val = (r.denominator + r.numerator - 1) // r.numerator
            if d_val in used_denoms or d_val in vals or d_val > 10000000:
                # Try next possible
                found = False
                for d_try in range(d_val + 1, min(d_val + 100, 10000001)):
                    if d_try not in used_denoms and d_try not in vals:
                        new_r = r - Fraction(1, d_try)
                        if new_r > 0:
                            vals.append(d_try)
                            r = new_r
                            found = True
                            break
                if not found:
                    return None
            else:
                vals.append(d_val)
                r -= Fraction(1, d_val)
        
        if r == 0 and len(vals) <= max_terms:
            return vals
        return None

    # Phase 1: Exhaustive 3-term search
    for a in range(7, 19):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        
        pairs = solve_2term(r)
        for b, c in pairs:
            if b != a and c != a and b != c:
                if b <= 10000000 and c <= 10000000:
                    check_solution([a, b, c])

    # Phase 2: Modular arithmetic approach for 4+ terms
    # We want 1/a + 1/b + ... = 5/31
    # For small a, b, solve for remainder and complete
    
    # Generate candidate (a, b) pairs where a < b
    # 1/a + 1/b < 5/31 => (a+b)/(ab) < 5/31 => 31(a+b) < 5ab
    # Also 1/a + 1/b > 0 and remainder must be positive and completable
    
    a_min = 7  # 1/a < 5/31 => a > 31/5 = 6.2
    a_max = 24  # 4/a > 5/31 => a < 124/5 = 24.8 for 4-term
    
    for a in range(a_min, a_max + 1):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        
        # For each b > a, check if remainder is positive and completable
        # 1/b < 5/31 - 1/a => b > 1/(5/31 - 1/a)
        r_after_a = target - Fraction(1, a)
        if r_after_a <= 0:
            continue
        
        # b must satisfy: 1/b < r_after_a and 1/b > 0
        # Also need r_after_a - 1/b > 0 for at least 2 more terms
        # So 1/b < r_after_a => b > 1/r_after_a
        b_lo = max(a + 1, r_after_a.denominator // r_after_a.numerator + 1)
        
        # Upper bound: we need at least 2 more terms after b
        # 2/c > r_after_a - 1/b => c < 2/(r_after_a - 1/b)
        # For feasibility, r_after_a - 1/b must be positive and not too small
        # Loose upper bound: b < 1/(r_after_a/2) approximately
        b_hi = min(best_max_den - 1, (2 * r_after_a.denominator) // r_after_a.numerator)
        
        if b_lo > b_hi:
            continue
        
        for b in range(b_lo, b_hi + 1):
            if time.time() - start_time > DEADLINE:
                break
            if b >= best_max_den:
                break
            
            r_after_ab = r_after_a - Fraction(1, b)
            if r_after_ab <= 0:
                continue
            
            # Try to complete with minimal terms
            used = {a, b}
            completion = complete_with_min_terms(r_after_ab, used, max_terms=254)
            if completion is not None:
                total = [a, b] + completion
                if len(total) <= 256:
                    check_solution(total)

    if best_solution is not None:
        return best_solution

    # Fallback: Greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
