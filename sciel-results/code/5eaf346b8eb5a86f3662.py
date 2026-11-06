"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_complete(r, start_d, max_denom, used_set, max_terms=256):
    """Complete r using greedy, all denominators >= start_d, <= max_denom, distinct."""
    values = []
    r = Fraction(r)
    d_min = start_d
    while r > 0:
        if len(values) >= max_terms:
            return None
        if r.numerator == 0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d < d_min:
            d = d_min
        if d > max_denom:
            return None
        if d in used_set:
            d += 1
            while d <= max_denom and d in used_set:
                d += 1
            if d > max_denom:
                return None
        values.append(d)
        used_set.add(d)
        r -= Fraction(1, d)
        d_min = d + 1
    return values

def _dfs_complete(r, start_d, max_denom, used_set, depth, max_depth, max_terms, deadline):
    """Bounded DFS to complete r. Returns list of denominators or None.
    Only explores denominators <= max_denom, distinct, >= start_d."""
    if time.time() > deadline:
        return None
    if r == 0:
        return []
    if depth >= max_depth:
        # Fall back to greedy for the remainder
        return _greedy_complete(r, start_d, max_denom, used_set, max_terms)
    if r.numerator == 0:
        return []

    # 1/d <= r => d >= ceil(1/r) = ceil(r.denominator / r.numerator)
    d_lo = (r.denominator + r.numerator - 1) // r.numerator
    if d_lo < start_d:
        d_lo = start_d
    if d_lo > max_denom:
        return None

    # We also need enough remaining terms: if we have max_terms - len(used) - 1 left...
    # Just bound the search: only try a limited number of candidate d's
    # Try a window of candidate denominators starting from d_lo
    # To keep it bounded, try at most `window` values
    window = 30
    d_hi = min(d_lo + window, max_denom)

    for d in range(d_lo, d_hi + 1):
        if d in used_set:
            continue
        new_used = used_set | {d}
        new_r = r - Fraction(1, d)
        if new_r < 0:
            continue
        # remaining terms available
        remaining_terms = max_terms - len(used_set) - 1
        if remaining_terms <= 0:
            continue
        res = _dfs_complete(new_r, d + 1, max_denom, new_used,
                            depth + 1, max_depth, remaining_terms, deadline)
        if res is not None:
            return [d] + res
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')

    # Minimum possible denominator: 1/d <= 5/31 => d >= ceil(31/5) = 7
    min_d = 7
    max_d_upper = 10000000

    # Phase 1: Greedy with various starting denominators
    for first_d in range(min_d, min_d + 5000):
        if time.time() > deadline:
            break
        remainder = target - Fraction(1, first_d)
        if remainder < 0:
            break
        used = {first_d}
        completion = _greedy_complete(remainder, first_d + 1, max_d_upper, used)
        if completion is not None:
            all_denoms = [first_d] + completion
            max_denom = max(all_denoms)
            if max_denom < best_max_denom:
                best_max_denom = max_denom
                best_solution = all_denoms

    # Phase 2: DFS refinement around best first_d to beat current best
    if best_solution is not None:
        best_first = best_solution[0]
        search_range = range(max(min_d, best_first - 30), best_first + 30)
        for first_d in search_range:
            if time.time() > deadline:
                break
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                continue
            used = {first_d}
            max_allowed = best_max_denom - 1
            if max_allowed < first_d + 1:
                continue
            # First try greedy with the cap
            completion = _greedy_complete(remainder, first_d + 1, max_allowed, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms
            # Then try bounded DFS to escape greedy traps
            if time.time() < deadline:
                completion = _dfs_complete(remainder, first_d + 1, max_allowed,
                                           used, 0, 3, 256, deadline)
                if completion is not None:
                    all_denoms = [first_d] + completion
                    max_denom = max(all_denoms)
                    if max_denom < best_max_denom:
                        best_max_denom = max_denom
                        best_solution = all_denoms

    # Phase 3: Random exploration if time remains
    if time.time() < deadline - 3.0:
        import random
        random.seed(42)
        for _ in range(20000):
            if time.time() > deadline:
                break
            first_d = random.randint(min_d, min_d + 50000)
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                continue
            used = {first_d}
            completion = _greedy_complete(remainder, first_d + 1, best_max_denom - 1, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms

    if best_solution is None:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values

    return best_solution
# EVOLVE-BLOCK-END
