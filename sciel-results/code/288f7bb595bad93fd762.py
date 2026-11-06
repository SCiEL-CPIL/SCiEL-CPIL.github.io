import math
import time

# Precompute 12!
T12 = math.factorial(12)
PRIMES = (2, 3, 5, 7, 11)
TARGET = (10, 5, 2, 1, 1)
MAX_FACTORS = 12


def _build_candidates(bound):
    """Build candidate exponent vectors for divisors d with 12 < d <= bound."""
    cands = []
    for e2 in range(11):
        v2 = 1 << e2
        for e3 in range(6):
            v3 = v2 * (3 ** e3)
            for e5 in range(3):
                v5 = v3 * (5 ** e5)
                for e7 in range(2):
                    v7 = v5 * (7 ** e7)
                    for e11 in range(2):
                        d = v7 * (11 ** e11)
                        if 12 < d <= bound:
                            cands.append(((e2, e3, e5, e7, e11), d))
    # Sort by divisor value (for canonical non-decreasing ordering)
    cands.sort(key=lambda x: x[1])
    return cands


def _can_form(rem, cands, start_idx, bound):
    """Quick feasibility: can the remaining exponent vector 'rem' be formed by
    distinct candidates (indices >= start_idx) with values <= bound?
    Uses a necessary condition: rem must be componentwise >= 0 and the product
    of remaining must divide the product of all remaining candidates.
    We use a simple upper-bound check: the smallest product of any subset of
    size >= 1 must be <= rem_product. Actually just check rem is non-negative
    and that rem_product <= product of all remaining candidate values (loose)."""
    for r in rem:
        if r < 0:
            return False
    # rem must be formable: at minimum, rem_product must be >= 1 and the
    # largest single exponent in rem must not exceed the max available.
    # This is a weak check but cheap.
    return True


def _search(rem, cands, start_idx, bound, chosen, remaining_slots):
    """Backtracking: find distinct candidates (values non-decreasing) whose
    exponent vectors sum to rem, using at most remaining_slots factors.
    Returns list of chosen divisors or None."""
    # Check if rem is zero -> success
    if all(r == 0 for r in rem):
        return list(chosen)

    # We need at least 1 more factor; at most remaining_slots
    if remaining_slots <= 0:
        return None

    # The product of remaining must be formable. Compute rem_product.
    rem_product = 1
    for r, p in zip(rem, PRIMES):
        rem_product *= p ** r

    n = len(cands)
    for i in range(start_idx, n):
        exp, d = cands[i]
        # Non-decreasing ordering is automatic since cands sorted by d and i >= start_idx
        # But we must ensure d <= bound (already filtered) and d > 12 (already filtered)
        # Check that choosing d doesn't exceed rem in any component
        if any(exp[k] > rem[k] for k in range(5)):
            continue
        # Pruning: after choosing d, the remaining product is rem_product // d
        # We need at least 1 more factor (unless rem becomes 0)
        new_rem = tuple(rem[k] - exp[k] for k in range(5))
        if all(r == 0 for r in new_rem):
            # Success with exactly one more factor
            chosen.append(d)
            res = list(chosen)
            chosen.pop()
            return res

        # Need at least 1 more factor after this one
        if remaining_slots - 1 >= 1:
            # Pruning: the next factor must be >= d (non-decreasing) and <= bound
            # Also, the remaining product must be >= the smallest possible next factor
            # and the remaining product must be formable by candidates from i+1 onward
            # Quick check: is there any candidate j >= i+1 with value <= bound
            # whose exp is componentwise <= new_rem?
            has_next = False
            for j in range(i + 1, n):
                exp_j, d_j = cands[j]
                if d_j > bound:
                    break
                if all(exp_j[k] <= new_rem[k] for k in range(5)):
                    has_next = True
                    break
            if not has_next:
                continue

            chosen.append(d)
            res = _search(new_rem, cands, i + 1, bound, chosen, remaining_slots - 1)
            chosen.pop()
            if res is not None:
                return res
        # If remaining_slots - 1 == 0, we can't add more, so skip (handled above by the all-zero check)

    return None


def _find_decomposition(bound):
    """Try to find a decomposition with all factors <= bound. Returns list or None."""
    cands = _build_candidates(bound)
    if not cands:
        return None
    # Try with up to MAX_FACTORS factors, starting from fewer (fewer factors
    # tends to give smaller max, but we're searching for feasibility at this bound)
    # Actually, for a given bound, we just need ANY valid decomposition.
    # Start with max slots = min(MAX_FACTORS, len(cands))
    max_slots = min(MAX_FACTORS, len(cands))
    for slots in range(1, max_slots + 1):
        res = _search(TARGET, cands, 0, bound, [], slots)
        if res is not None:
            return res
    return None


def _two_factor_decomp():
    """Find best 2-factor decomposition."""
    best = None
    sqrt_t = int(math.isqrt(T12))
    for d in range(13, sqrt_t + 1):
        if T12 % d == 0:
            other = T12 // d
            if other > 12 and d != other:
                if best is None or max(d, other) < max(best):
                    best = (d, other)
    return best


def _three_factor_decomp():
    """Find best 3-factor decomposition with all factors > 12 and distinct."""
    best = None
    # f1 <= f2 <= f3, f1*f2*f3 = T12, all > 12, all distinct
    f1_max = int(T12 ** (1/3)) + 1
    for f1 in range(13, f1_max + 1):
        if T12 % f1 != 0:
            continue
        rem = T12 // f1
        f2_max = int(math.isqrt(rem)) + 1
        for f2 in range(max(13, f1 + 1), f2_max + 1):
            if rem % f2 != 0:
                continue
            f3 = rem // f2
            if f3 <= f2:
                continue
            if f3 > 12 and f3 != f2 and f3 != f1:
                if best is None or max(f1, f2, f3) < max(best):
                    best = (f1, f2, f3)
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Get initial bounds from small-factor decompositions
    two = _two_factor_decomp()
    three = _three_factor_decomp()

    candidates = []
    if two:
        candidates.append(two)
    if three:
        candidates.append(three)

    best_solution = None
    if candidates:
        best_solution = min(candidates, key=lambda x: max(x))
    else:
        best_solution = (T12,)

    current_bound = max(best_solution)

    # Binary search on the bound
    lo = 13
    hi = current_bound

    # First, try to find the optimal with a reasonable upper bound
    # The optimal is likely small. Let's do a linear scan from a small start
    # up to current_bound, but that could be many iterations.
    # Better: binary search.

    # But binary search requires monotonicity: if we can form with bound B,
    # we can form with bound B+1 (same decomposition works). So it's monotone.
    # We want the smallest B such that a decomposition exists.

    # However, the number of factors is variable, so a decomposition with
    # bound B might use more factors than one with bound B+1. But feasibility
    # is still monotone: if feasible at B, feasible at B+1.

    # Let's binary search.
    found_lo = lo  # We know it's feasible at current_bound
    found_hi = hi

    # First check if it's feasible at a small bound
    # Try bounds: 13, 14, 15, ... up to some point, then binary search
    # Actually, let's just binary search directly.

    while lo <= hi and (time.time() - start_time) < 80:
        mid = (lo + hi) // 2
        res = _find_decomposition(mid)
        if res is not None:
            # Found a decomposition with max factor <= mid
            # Update best if mid < current_bound
            if mid < current_bound:
                current_bound = mid
                best_solution = sorted(res)
            hi = mid - 1
        else:
            lo = mid + 1

    # Verify the best solution
    if best_solution:
        prod = 1
        for d in best_solution:
            prod *= d
        if (prod == T12 and 1 <= len(best_solution) <= 12 and
                all(d > 12 for d in best_solution) and
                len(set(best_solution)) == len(best_solution)):
            return sorted(best_solution)

    # Fallback
    return [T12]


if __name__ == "__main__":
    res = construct()
    print(res)
    print(f"Max factor: {max(res) if res else 'N/A'}")
    print(f"Product: {math.prod(res) if res else 'N/A'}")
