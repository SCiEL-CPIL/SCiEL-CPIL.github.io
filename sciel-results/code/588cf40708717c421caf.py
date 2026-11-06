def construct():
    import time
    from fractions import Fraction

    target = Fraction(5, 31)
    best_max = float('inf')
    best_denoms = []

    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s kill

    # Binary search for the minimum maximum denominator K
    # such that there exists an Egyptian fraction expansion of 5/31
    # with all denominators <= K.

    # First, find an upper bound using greedy from d1=8
    def greedy_complete(r, used, K):
        """Complete the expansion using greedy method, ensuring all denoms <= K."""
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
                # Try next few denominators
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

    # Get a baseline solution to establish upper bound
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

    # Lower bound: smallest possible max denominator
    # The first denominator must be > 31/5 = 6.2, so at least 7
    # If we use just 1/7, remainder is 5/31 - 1/7 = (35-31)/217 = 4/217
    # So lower bound is at least 7
    lower_bound = 7

    # Binary search for the minimum K
    lo = lower_bound
    hi = upper_bound

    # Try to find the minimum K using binary search
    # For each K, try different first denominators and see if we can complete
    def can_complete_with_bound(K):
        """Try to find an Egyptian fraction expansion of 5/31 with all denoms <= K."""
        for d1 in range(7, min(K + 1, 100)):
            if time.time() > deadline:
                return None
            r = target - Fraction(1, d1)
            if r <= 0:
                continue
            if d1 > K:
                break
            
            used = {d1}
            denoms = [d1]
            
            # Try to complete with greedy, bounded by K
            completion = greedy_complete(r, used, K)
            if completion is not None:
                solution = [d1] + completion
                if max(solution) <= K:
                    return solution
        
        return None

    # Binary search
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        solution = can_complete_with_bound(mid)
        if solution is not None:
            hi = mid
            if max(solution) < best_max:
                best_max = max(solution)
                best_denoms = solution[:]
        else:
            lo = mid + 1

    # Try to improve with the found best_max
    # Also try values slightly above best_max in case binary search missed something
    for K in range(best_max, min(best_max + 5, upper_bound + 1)):
        if time.time() > deadline:
            break
        solution = can_complete_with_bound(K)
        if solution is not None:
            if max(solution) < best_max:
                best_max = max(solution)
                best_denoms = solution[:]

    # If no improvement found, return the baseline
    if not best_denoms:
        best_denoms = baseline

    return best_denoms
