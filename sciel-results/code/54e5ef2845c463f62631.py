def construct():
    import time
    from fractions import Fraction

    target = Fraction(5, 31)
    best_max = float('inf')
    best_denoms = []

    start_time = time.time()
    deadline = start_time + 55.0

    def max_sum_recip(d, k):
        s = Fraction(0, 1)
        for i in range(k):
            s += Fraction(1, d + i)
        return s

    def min_sum_recip(max_d, k):
        s = Fraction(0, 1)
        for i in range(k):
            s += Fraction(1, max_d - i)
        return s

    def dfs(rem, min_d, max_d, depth, path):
        if depth == 0:
            return rem == 0
        if rem <= 0:
            return False
        
        # Pruning
        if max_sum_recip(min_d, depth) < rem:
            return False
        if min_sum_recip(max_d, depth) > rem:
            return False

        d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        d_max = max_d
        
        for d in range(d_min, d_max + 1):
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            
            if depth - 1 == 0:
                if new_rem == 0:
                    path.append(d)
                    return True
            else:
                if max_sum_recip(d + 1, depth - 1) < new_rem:
                    break
                if min_sum_recip(max_d, depth - 1) > new_rem:
                    break
            
            path.append(d)
            if dfs(new_rem, d + 1, max_d, depth - 1, path):
                return True
            path.pop()
            
        return False

    def find_solution(K, max_terms=10):
        for depth in range(2, max_terms + 1):
            path = []
            if dfs(target, 1, K, depth, path):
                return path
        return None

    # Get an upper bound using greedy
    def greedy_complete(r, used, K):
        if r == 0:
            return []
        if r < 0:
            return None
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > K:
                return None
            if d in used:
                found = False
                for dd in range(d, min(d + 100, K + 1)):
                    if dd not in used and dd <= K:
                        new_r = r - Fraction(1, dd)
                        if 0 <= new_r:
                            values.append(dd)
                            used.add(dd)
                            r = new_r
                            found = True
                            break
                if not found:
                    return None
            else:
                values.append(d)
                used.add(d)
                r -= Fraction(1, d)
        return values

    # Baseline
    baseline = [8]
    r = target - Fraction(1, 8)
    used = {8}
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        baseline.append(d)
        used.add(d)
        r -= Fraction(1, d)
    
    upper_bound = max(baseline)
    best_max = upper_bound
    best_denoms = baseline[:]

    # Binary search for minimum K
    lo = 7
    hi = upper_bound

    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = find_solution(mid)
        if sol is not None:
            hi = mid
            if max(sol) < best_max:
                best_max = max(sol)
                best_denoms = sol[:]
        else:
            lo = mid + 1

    # Verify best
    if best_denoms:
        s = sum(Fraction(1, d) for d in best_denoms)
        if s == target:
            return best_denoms

    return baseline
