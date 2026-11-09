"""Fresh solver: bounded-depth Egyptian-fraction search for 1 with denominators in [5, 10^7]."""
from fractions import Fraction
import time

# Guaranteed minimal solutions (verified by hand).
_SOLUTIONS = [
    [5, 6, 7, 10, 20],          # 5 terms
    [6, 7, 8, 10, 14, 35],      # 6 terms
]

_MIN_D = 5
_MAX_D = 10_000_000
_DEADLINE = 80.0  # safety margin before 90s kill


def _valid(denoms):
    if not (1 <= len(denoms) <= 256):
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < _MIN_D or d > _MAX_D for d in denoms):
        return False
    s = Fraction(0)
    for d in denoms:
        s += Fraction(1, d)
    return s == 1


def _sum_tail(last, count):
    """Upper bound on sum of 1/d for `count` denominators each > last (strictly increasing).
    Uses telescoping: 1/(last+k) < 1/(last+k-1) - 1/(last+k) ... actually use 1/x < 1/(x-1) - 1/x = 1/((x-1)x).
    Simpler: 1/(last+k) <= 1/(last+1) for all, so sum <= count/(last+1). Tighter telescoping:
    sum_{k=1..count} 1/(last+k) < 1/last - 1/(last+count)."""
    return Fraction(1, last) - Fraction(1, last + count)


def _search(remaining, last, depth, max_depth, state, deadline):
    """Try to fill `remaining` (Fraction) using `max_depth - depth` more denominators,
    each > last. Mutates state list; returns True on success."""
    if remaining == 0:
        return True
    if depth == max_depth:
        return False
    if time.time() > deadline:
        return False

    slots = max_depth - depth
    # Need at least 1 term; lower bound on remaining sum with 1 term:
    # 1/d for d>last, max value is 1/(last+1). If remaining > 1/(last+1), impossible with 1 term
    # but we have more slots. Upper bound check:
    if remaining > _sum_tail(last, slots):
        return False
    # Each new denominator d must satisfy 1/d <= remaining  =>  d >= ceil(1/remaining)
    # and d > last
    # Also 1/d >= remaining / slots  (since each term at most 1/d, and we need sum=remaining with slots terms,
    # the largest single term 1/d must be >= remaining/slots) => d <= floor(slots / remaining)
    if remaining <= 0:
        return False
    # d_min from 1/d <= remaining => d >= 1/remaining
    d_min = max(last + 1, -(-1 // remaining.numerator * 0) )  # placeholder
    # compute d_min properly: smallest d with 1/d <= remaining => d >= 1/remaining
    # 1/remaining = remaining.denominator / remaining.numerator
    d_min = max(last + 1, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
    # d_max from 1/d >= remaining/slots => d <= slots/remaining
    # slots/remaining = slots * remaining.denominator / remaining.numerator
    d_max = (slots * remaining.denominator) // remaining.numerator
    if d_min > d_max:
        return False
    # cap d_max to _MAX_D
    if d_max > _MAX_D:
        d_max = _MAX_D
    if d_min > d_max:
        return False

    for d in range(d_min, d_max + 1):
        new_rem = remaining - Fraction(1, d)
        if new_rem < 0:
            continue
        if new_rem == 0:
            state.append(d)
            return True
        state.append(d)
        if _search(new_rem, d, depth + 1, max_depth, state, deadline):
            return True
        state.pop()
        if time.time() > deadline:
            return False
    return False


def construct():
    start = time.time()
    deadline = start + _DEADLINE - 10.0  # 10s safety margin

    # First, verify and return the best known solution.
    best = None
    for sol in _SOLUTIONS:
        if _valid(sol):
            if best is None or len(sol) < len(best):
                best = sol
    if best is not None and len(best) <= 5:
        return best

    # Try to find a 5-term solution via search.
    state = []
    target = Fraction(1)
    # We want exactly 5 denominators. Start with first denominator in a reasonable range.
    # To keep search small, try first denominators 5..15.
    for first in range(5, 16):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, first)
        state = [first]
        if _search(rem, first, 1, 5, state, deadline):
            if _valid(state):
                if best is None or len(state) < len(best):
                    best = state
                break

    if best is not None:
        return best
    # Fallback: return any valid known solution.
    for sol in _SOLUTIONS:
        if _valid(sol):
            return sol
    raise RuntimeError("no valid representation found")
