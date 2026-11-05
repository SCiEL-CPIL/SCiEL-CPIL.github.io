def construct():
    import time
    start_time = time.time()
    DEADLINE = 85

    from fractions import Fraction

    target = Fraction(5, 31)
    best_max_den = 10000001
    best_solution = None

    def check_solution(denoms):
        nonlocal best_max_den, best_solution
        if not denoms or len(denoms) > 256:
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

    def solve_2term_fast(r):
        """Solve 1/b + 1/c = r where r = n/d reduced. Returns list of (b,c) with b<=c."""
        n = r.numerator
        d = r.denominator
        if n == 0:
            return []
        d2 = d * d
        results = []
        # (b*n - d)(c*n - d) = d2
        # Let u = b*n - d, v = c*n - d, u*v = d2
        # b = (u+d)/n, c = (v+d)/n
        # Need u+d > 0 and v+d > 0, and both divisible by n
        from math import isqrt
        limit = isqrt(d2)
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu <= 0 or bv <= 0:
                continue
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                if b > 0 and c > 0:
                    if b > c:
                        b, c = c, b
                    results.append((b, c))
        return results

    def dfs(r, min_den, used, remaining_terms, current_denoms):
        """DFS to find Egyptian fraction for r with denominators >= min_den, distinct from used."""
        nonlocal best_max_den, best_solution
        if time.time() - start_time > DEADLINE:
            return False
        if r == 0:
            if current_denoms:
                check_solution(current_denoms)
            return True
        if r < 0:
            return False
        if remaining_terms == 0:
            return False
        if remaining_terms == 1:
            # Need 1/x = r, so x = 1/r
            if r.numerator == 1:
                x = r.denominator
                if x >= min_den and x not in used and x <= 10000000:
                    check_solution(current_denoms + [x])
            return False
        if remaining_terms == 2:
            pairs = solve_2term_fast(r)
            for (b, c) in pairs:
                if b >= min_den and b not in used and c not in used and b != c:
                    if b <= 10000000 and c <= 10000000:
                        check_solution(current_denoms + [b, c])
            return False

        # For remaining_terms >= 3
        # Lower bound for next denominator: 1/d <= r => d >= ceil(1/r)
        # Also d >= min_den
        # Upper bound: we need remaining_terms denominators each >= d
        # So r >= remaining_terms / d_max_possible... 
        # Actually, 1/d + 1/(d+1) + ... must be >= r
        # Simplest: r >= remaining_terms / (d + remaining_terms - 1) approximately
        # Better: r >= remaining_terms / d (since all denominators >= d)
        # So d >= remaining_terms / r... wait no.
        # 1/d + 1/(d+1) + ... + 1/(d+k-1) >= r
        # Since each term <= 1/d, sum <= k/d, so we need k/d >= r, i.e., d <= k/r
        # Wait, that's the upper bound on d for feasibility.
        # Lower bound: 1/d <= r => d >= ceil(1/r)
        
        d_lo = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        
        # Upper bound: we need sum of remaining_terms reciprocals >= r
        # Each reciprocal <= 1/d_lo... no, each >= 1/d where d is the current one
        # Actually, the maximum sum with remaining_terms terms starting from d is:
        # 1/d + 1/(d+1) + ... which is bounded by remaining_terms/d
        # So we need remaining_terms/d >= r, i.e., d <= remaining_terms/r
        d_hi = (remaining_terms * r.denominator) // r.numerator
        if d_hi < d_lo:
            return False
        
        # Also, if we've found a solution, we can prune: max so far + new terms
        # If current max + something > best_max_den, skip
        
        for d in range(d_lo, d_hi + 1):
            if time.time() - start_time > DEADLINE:
                return False
            if d in used:
                continue
            if d > best_max_den - 1:
                break
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break  # d too small, but as d increases, 1/d decreases, so new_r increases
                        # Wait, new_r = r - 1/d. As d increases, 1/d decreases, new_r increases.
                        # So if new_r < 0, it will stay < 0? No, it increases toward r.
                        # Actually if new_r < 0, that means 1/d > r, so d < 1/r.
                        # But d_lo = ceil(1/r), so this shouldn't happen.
                        # Unless there's an off-by-one. Let's just continue.
                continue
            if new_r == 0:
                check_solution(current_denoms + [d])
                continue
            dfs(new_r, d + 1, used | {d}, remaining_terms - 1, current_denoms + [d])

    # Try different numbers of terms
    for num_terms in range(2, 8):
        if time.time() - start_time > DEADLINE:
            break
        dfs(target, 1, set(), num_terms, [])
        if best_solution is not None and best_max_den <= 10:
            break

    if best_solution is not None:
        return best_solution

    # Fallback: greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
