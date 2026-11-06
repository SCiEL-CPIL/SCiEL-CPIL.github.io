"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

_MAX_DEN = 10000000


def _max_fill_float(last, count):
    """Approximate max reciprocal sum with `count` distinct denominators > last."""
    if count <= 0:
        return 0.0
    s = 0.0
    for j in range(1, count + 1):
        s += 1.0 / (last + j)
    return s


def _min_fill_float(last, count):
    """Approximate min reciprocal sum with `count` distinct denominators > last (largest possible)."""
    if count <= 0:
        return 0.0
    s = 0.0
    for j in range(1, count + 1):
        s += 1.0 / (_MAX_DEN - count + j)
    return s


def _greedy_egyptian(remaining, last, deadline):
    """Greedy: pick smallest valid d > last such that 1/d <= remaining and
       remaining - 1/d is still representable with remaining-1 terms."""
    if remaining == 0:
        return None
    if remaining == 1:
        # remaining must be exactly 1/d for some valid d
        # remaining is a Fraction
        if remaining.denominator == 1 and remaining.numerator == 1:
            # Need 1/d = 1, so d=1, but d>=5 required
            return None
        # 1/d = remaining => d = 1/remaining = remaining.denominator/remaining.numerator
        d = remaining.denominator // remaining.numerator
        if remaining.numerator * d != remaining.denominator:
            return None
        if d > last and d >= 5 and d <= _MAX_DEN:
            return [d]
        return None

    # For remaining > 1, find smallest d > last such that:
    # 1. 1/d <= remaining  => d >= 1/remaining
    # 2. remaining - 1/d >= min_fill_float(d, remaining-1)  (can still reach 0 with remaining-1 terms)
    # 3. remaining - 1/d <= max_fill_float(d, remaining-1)  (can still reach target)
    # 4. d >= 5, d <= _MAX_DEN
    
    # Lower bound on d from 1/d <= remaining
    # d >= 1/remaining = remaining.denominator / remaining.numerator
    d_low = max(last + 1, 5)
    # Also need d >= ceil(1/remaining)
    d_from_frac = remaining.denominator // remaining.numerator
    if d_from_frac * remaining.numerator < remaining.denominator:
        d_from_frac += 1
    d_low = max(d_low, d_from_frac)
    
    d = d_low
    guard = 0
    while d <= _MAX_DEN and guard < 500000:
        guard += 1
        if time.time() > deadline:
            return None
        
        # Check 1/d <= remaining
        if d * remaining.numerator > remaining.denominator:
            # 1/d > remaining, skip
            d += 1
            continue
        
        new_remaining = remaining - Fraction(1, d)
        
        # Check new_remaining >= 0
        if new_remaining < 0:
            d += 1
            continue
        
        # Check feasibility: can we represent new_remaining with (remaining-1) terms > d?
        max_possible = _max_fill_float(d, remaining - 1)
        if new_remaining > max_possible + 1e-15:
            # Even with smallest denominators, we can't reach new_remaining
            # Try larger d (smaller 1/d, larger new_remaining) - wait, that makes it harder
            # Actually as d increases, 1/d decreases, new_remaining increases
            # So if new_remaining > max_possible, larger d makes it worse
            # We need to try smaller d, but we're already at the minimum valid d
            # This means no solution with this branch
            return None
        
        min_possible = _min_fill_float(d, remaining - 1)
        if new_remaining < min_possible - 1e-15:
            # new_remaining is too small to be represented by remaining-1 terms > d
            # As d increases, 1/d decreases, new_remaining increases, so this might work
            d += 1
            continue
        
        # Also check: new_remaining must be representable, which requires
        # new_remaining >= sum of (remaining-1) smallest possible reciprocals > d
        # and new_remaining <= sum of (remaining-1) largest possible reciprocals > d
        # (which are 1/(d+1), 1/(d+2), ...)
        
        res = _greedy_egyptian(remaining - 1, d, deadline)
        if res is not None:
            return [d] + res
        
        d += 1
    
    return None


def _search(k, deadline):
    """Try to find exactly k distinct denominators >= 5 summing to 1 using greedy."""
    return _greedy_egyptian(k, 4, deadline)


def _fallback():
    """Guaranteed-valid baseline solution using subset DP over divisors of a common denominator."""
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and (total + w) not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    for target in [5040, 10080, 25200, 50400, 100800]:
        reached = {0: []}
        for d in range(5, min(target + 1, 100001)):
            if target % d != 0:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and (total + w) not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
    
    raise RuntimeError("no solution found")


def construct():
    deadline = time.time() + 55.0
    
    # Try greedy for increasing k starting from 8
    for k in range(8, 257):
        if time.time() > deadline:
            break
        sol = _search(k, deadline)
        if sol is not None:
            return sol
    
    return _fallback()
# EVOLVE-BLOCK-END
