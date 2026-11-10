"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin before 60s limit

    # The minimum possible k is 8 (sum of 1/5+...+1/12 > 1, but 1/5+...+1/11 < 1)
    # Try k=8 with DFS using tight bounds
    result = _search_k(8, start, deadline)
    if result is not None:
        return result

    # Try k=9
    result = _search_k(9, start, deadline)
    if result is not None:
        return result

    # Fallback: DP over divisors of 2520 (known to give a valid answer)
    return _dp_fallback()


def _search_k(k, start, deadline):
    """DFS for exactly k distinct denominators >= 5 summing to 1."""
    # Precompute harmonic sums for pruning
    # max_sum(d, r) = 1/d + 1/(d+1) + ... + 1/(d+r-1)
    # We need: target <= max_sum(d, r) and target >= min_sum(d, r)
    # min_sum for r terms from d: 1/d + 1/(d+1) + ... but that's the max
    # Actually for distinct increasing denominators starting at d:
    # max sum = 1/d + 1/(d+1) + ... + 1/(d+r-1)
    # min sum = 1/(d+r-1) + 1/(d+r) + ... no, that's wrong
    # The min sum of r distinct denominators all >= d is 1/(d) + 1/(d+1) + ... no
    # Wait, to minimize the sum we want the LARGEST denominators
    # But we don't have an upper bound easily. Let's use:
    # For the remaining r terms with min denominator d:
    # max possible sum = 1/d + 1/(d+1) + ... + 1/(d+r-1)
    # If target > max possible sum, prune
    # Also: each next denom must be > previous, so next denom >= d
    # target must be representable: 1/d_next >= target/r => d_next <= r/target
    
    def dfs(target, min_den, remaining, current):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current if target == 0 else None
        
        if target <= 0:
            return None
        
        # Bound: max sum of 'remaining' terms starting from min_den
        # = 1/min_den + 1/(min_den+1) + ... + 1/(min_den+remaining-1)
        max_sum = Fraction(0, 1)
        for i in range(remaining):
            max_sum += Fraction(1, min_den + i)
        if target > max_sum:
            return None
        
        # The next denominator d: 
        # We need 1/d + (max sum of remaining-1 from d+1) >= target
        # And 1/d <= target (since all terms positive and we need target - 1/d >= 0)
        # Also d >= min_den
        # d <= remaining / target (since 1/d >= target/remaining for feasibility)
        
        # Upper bound on d: 1/d must be at most target (trivial) but also
        # we need the remaining (remaining-1) terms to sum to target - 1/d
        # The max sum of (remaining-1) terms from d+1 is 1/(d+1)+...+1/(d+remaining-1)
        # So: target - 1/d <= 1/(d+1) + ... + 1/(d+remaining-1)
        
        # Simple upper bound: d <= remaining / target
        max_d = min(10000000, int(remaining / target) + 1) if target > 0 else 10000000
        if max_d < min_den:
            return None
        
        for d in range(min_den, max_d + 1):
            if time.time() > deadline:
                return None
            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            if new_target == 0:
                if remaining == 1:
                    return current + [d]
                else:
                    continue
            # Check: can (remaining-1) terms from d+1 sum to new_target?
            # Max sum check
            max_rem = Fraction(0, 1)
            for i in range(remaining - 1):
                max_rem += Fraction(1, d + 1 + i)
            if new_target > max_rem:
                continue
            # Also need new_target > 0 (already checked)
            result = dfs(new_target, d + 1, remaining - 1, current + [d])
            if result is not None:
                return result
        return None
    
    return dfs(Fraction(1, 1), 5, k, [])


def _dp_fallback():
    """Subset DP over divisors of 2520."""
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
