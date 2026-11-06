"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 12
    target = math.factorial(N)  # 479001600
    min_factor = N + 1  # 13

    # We want 1..12 DISTINCT integers all > 12, product = target
    # Minimize the largest factor.
    #
    # Strategy: DFS over divisors in increasing order with aggressive pruning.
    # State: (remaining_product, min_allowed_factor, current_factors_list, current_max)
    #
    # Key improvements:
    # 1. Precompute all divisors of target once.
    # 2. Use lru_cache for divisor enumeration of sub-products.
    # 3. Stronger pruning:
    #    - If current_max >= best, prune.
    #    - If remaining < min_allowed, prune.
    #    - If we can't form remaining with distinct factors >= min_allowed, prune.
    #    - Bound the number of additional factors needed.
    # 4. Better initial upper bound via greedy/heuristic.
    # 5. Iterate divisors in sorted order, break early when d >= best_largest.

    # Precompute all divisors of target
    all_divisors_target = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            all_divisors_target.append(i)
            if i != target // i:
                all_divisors_target.append(target // i)
    all_divisors_target.sort()
    # Filter to those > 12
    divisors_above_min = [d for d in all_divisors_target if d > 12]

    @lru_cache(maxsize=None)
    def get_divisors(n):
        """Return sorted list of divisors of n that are >= min_factor."""
        divs = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                if i >= min_factor:
                    divs.append(i)
                if i != n // i and n // i >= min_factor:
                    divs.append(n // i)
        divs.sort()
        return tuple(divs)

    # Initial upper bound: use a simple heuristic
    # Try to factor target into distinct factors > 12, greedily from smallest
    # This gives a rough upper bound on the largest factor
    best_largest = [target]  # worst case: single factor
    best_factors = [target]

    # Try a few heuristic starts to get a better initial upper bound
    # Heuristic: pick small divisors first
    def heuristic_init():
        """Try to find a valid decomposition greedily to set initial upper bound."""
        remaining = target
        factors = []
        min_allowed = min_factor
        while remaining > 1 and len(factors) < 12:
            # Find smallest divisor of remaining that is >= min_allowed
            found = None
            for d in get_divisors(remaining):
                if d >= min_allowed:
                    found = d
                    break
            if found is None:
                # Can't find a valid next factor; use remaining as last factor
                if remaining >= min_allowed:
                    factors.append(remaining)
                break
            factors.append(found)
            remaining //= found
            min_allowed = found + 1

        if len(factors) >= 1 and all(f > 12 for f in factors) and len(set(factors)) == len(factors):
            prod = 1
            for f in factors:
                prod *= f
            if prod == target:
                return max(factors), factors
        return None, None

    # Get initial bound from heuristic
    h_max, h_factors = heuristic_init()
    if h_factors:
        best_largest[0] = h_max
        best_factors[:] = h_factors[:]

    def dfs(remaining, min_allowed, factors_so_far, current_max):
        if time.time() > deadline:
            return

        # Prune: if current_max already >= best, no improvement possible
        if current_max >= best_largest[0]:
            return

        # Base case
        if remaining == 1:
            if factors_so_far:
                if current_max < best_largest[0]:
                    best_largest[0] = current_max
                    best_factors[:] = factors_so_far[:]
            return

        # If we've used 12 factors already and remaining != 1, dead end
        if len(factors_so_far) >= 12:
            return

        # Quick prune: remaining must be >= min_allowed (at least one more factor needed)
        if remaining < min_allowed:
            return

        # Find divisors of remaining that are >= min_allowed
        divs = get_divisors(remaining)

        for d in divs:
            if d < min_allowed:
                continue
            if d >= best_largest[0]:
                # Since divisors are sorted, all subsequent will be >= best
                break

            quotient = remaining // d

            # If quotient == 1, we're done
            if quotient == 1:
                new_max = max(current_max, d)
                if new_max < best_largest[0]:
                    best_largest[0] = new_max
                    best_factors[:] = factors_so_far + [d]
                continue

            # Lower bound check: quotient must be >= min_allowed_next
            min_allowed_next = d + 1
            if quotient < min_allowed_next:
                continue

            # Additional prune: check if quotient can be formed by distinct factors
            # all >= min_allowed_next. At minimum, quotient must have a divisor >= min_allowed_next.
            # We'll let recursion handle this, but add a quick check.

            new_max = max(current_max, d)
            if new_max >= best_largest[0]:
                continue

            factors_so_far.append(d)
            dfs(quotient, min_allowed_next, factors_so_far, new_max)
            factors_so_far.pop()

    # Start DFS
    dfs(target, min_factor, [], 0)

    # Ensure we return a valid result
    if best_factors:
        return best_factors[:]
    else:
        # Fallback: return target as single factor (shouldn't happen, but safety)
        return [target]
# EVOLVE-BLOCK-END
