"""
Different strategy: Use scipy.optimize.brentq as the primary engine, with a
lightweight bracketing phase.  The current program implements its own Newton/
Brent hybrids from scratch; this version delegates the heavy lifting to a
battle-tested, highly optimized C-level routine, and only spends its call
budget on finding a sign-changing bracket.  This is a materially different
design family (library-driven vs. hand-rolled iterative methods).
"""
import math
import time
from scipy.optimize import brentq

# Global deadline: evaluator kills after 150s wall-clock.
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490  # stay under 500


def _time_left():
    return _DEADLINE - time.time()


def _counted_call(f, x, counter):
    """Call f(x), track count, guard against exceptions / non-finite."""
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val


def _bracket(f, x0, counter, max_expansions=40):
    """
    Try to find a bracket [a, b] with opposite signs around x0.
    Uses exponential expansion in both directions.
    Returns (a, b) or None.
    """
    x = float(x0)
    f0 = _counted_call(f, x, counter)
    if not math.isfinite(f0):
        return None
    if abs(f0) <= 1e-10:
        return (x, x)

    step = 0.1
    for i in range(max_expansions):
        if _time_left() < 1.0:
            return None
        if counter[0] >= _CALL_BUDGET - 5:
            return None

        a = x - step
        b = x + step

        fa = _counted_call(f, a, counter)
        if not math.isfinite(fa):
            step *= 2
            continue
        fb = _counted_call(f, b, counter)
        if not math.isfinite(fb):
            step *= 2
            continue

        if fa * f0 < 0:
            return (a, x)
        if fb * f0 < 0:
            return (x, b)

        # Also check if a and b bracket each other (root between them, not at x)
        if fa * fb < 0:
            return (a, b)

        step *= 2
        if step > 1e6:
            break

    return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: bracket around x0, then call scipy.optimize.brentq.
    If bracketing fails, fall back to a small grid scan + brentq.
    """
    counter = [0]
    x0 = float(x0)

    # Primary: find a bracket and use brentq
    bracket = _bracket(f, x0, counter)
    if bracket is not None:
        a, b = bracket
        if a != b:
            try:
                result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                if abs(f(result)) <= 1e-8:
                    return result
            except Exception:
                pass

    # Fallback: small grid scan around x0 to find any sign change
    if _time_left() > 1.0 and counter[0] < _CALL_BUDGET - 10:
        # Try a set of offsets
        offsets = [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
                   0.001, -0.001, 0.5, -0.5, 100.0, -100.0]
        for delta in offsets:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
            a = x0 + delta
            b = x0
            fa = _counted_call(f, a, counter)
            fb = _counted_call(f, b, counter)
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
                try:
                    result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                    if abs(f(result)) <= 1e-8:
                        return result
                except Exception:
                    pass
            # Also try (b, x0 - delta)
            a2 = x0 - delta
            b2 = x0
            fa2 = _counted_call(f, a2, counter)
            if math.isfinite(fa2) and math.isfinite(fb) and fa2 * fb < 0:
                try:
                    result = brentq(f, a2, b2, xtol=1e-12, rtol=1e-12, maxiter=200)
                    if abs(f(result)) <= 1e-8:
                        return result
                except Exception:
                    pass

    # Last resort: if we haven't found anything, return x0
    return x0
