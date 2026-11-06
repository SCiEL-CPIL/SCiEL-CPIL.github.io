from fractions import Fraction
import time
from math import gcd

_DEADLINE = 0
_HARMONIC_CACHE = {}

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55

    # Try to find a solution with few terms first
    for k in range(8, 50):
        if time.time() > _DEADLINE:
            break
        result = _find_k_terms(k)
        if result:
            return result

    # Fallback: return the known 252-term solution [5..256]
    return list(range(5, 257))

def _verify_solution(denoms):
    if not denoms or len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    for d in denoms:
        if d < 5 or d > 10000000:
            return False
    total = Fraction(0, 1)
    for d in denoms:
        total += Fraction(1, d)
    return total == Fraction(1, 1)

def _harmonic_sum(start, count):
    """Compute sum of 1/(start+i) for i=0..count-1 using float with caching."""
    key = (start, count)
    if key in _HARMONIC_CACHE:
        return _HARMONIC_CACHE[key]
    s = 0.0
    for i in range(count):
        s += 1.0 / (start + i)
    _HARMONIC_CACHE[key] = s
    return s

def _find_k_terms(k):
    """Find k distinct denominators >= 5 summing to 1 using fast DFS."""
    global _DEADLINE

    def dfs(terms_rem, t_num, t_den, min_d, chosen):
        global _DEADLINE
        if time.time() > _DEADLINE:
            return None

        if terms_rem == 0:
            return chosen if t_num == 0 else None

        if terms_rem == 1:
            if t_num == 0:
                return None
            if t_den % t_num == 0:
                d = t_den // t_num
                if d >= min_d and d <= 10000000:
                    return chosen + [d]
            return None

        # Pruning: max sum with smallest denominators must be >= target
        max_s = _harmonic_sum(min_d, terms_rem)
        if max_s < t_num / t_den - 1e-15:
            return None

        # d bounds
        # Lower bound: 1/d <= t_num/t_den => d >= t_den/t_num
        # Also need: remaining terms_rem-1 terms starting from d+1 must be able to sum to (t_num/t_den - 1/d)
        # The max sum of terms_rem-1 terms starting from d+1 is harmonic_sum(d+1, terms_rem-1)
        # So we need: t_num/t_den - 1/d <= harmonic_sum(d+1, terms_rem-1)
        # This gives an upper bound on d

        d_lo = max(min_d, -(-t_den // t_num))  # ceil(t_den/t_num)

        # Upper bound: we need t_num/t_den - 1/d > 0, so d > t_den/t_num
        # Also, the remaining sum must be positive and achievable
        # For a tighter upper bound: even the smallest possible remaining sum
        # (using very large denominators) approaches 0, so d can be large
        # But we need at least terms_rem-1 distinct denominators >= d+1
        # The max contribution from remaining is harmonic_sum(d+1, terms_rem-1)
        # We need: t_num/t_den - 1/d <= harmonic_sum(d+1, terms_rem-1)

        # Simple upper bound: d <= terms_rem * t_den / t_num (each term at least 1/d)
        d_hi = min(10000000, (terms_rem * t_den) // t_num + 2)

        for d in range(d_lo, d_hi + 1):
            if time.time() > _DEADLINE:
                return None

            # Check: 1/d < t_num/t_den (strictly less, since we need positive remainder)
            if d * t_num <= t_den:
                continue

            new_num = t_num * d - t_den
            new_den = t_den * d

            # Check if remaining is achievable (float for speed)
            if terms_rem - 1 > 0:
                max_rest = _harmonic_sum(d + 1, terms_rem - 1)
                if new_num / new_den > max_rest + 1e-15:
                    break

            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g

            res = dfs(terms_rem - 1, new_num, new_den, d + 1, chosen + [d])
            if res is not None:
                return res

        return None

    return dfs(k, 1, 1, 5, [])
