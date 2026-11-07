"""Rewritten program: SEARCH-based root finder.
Generates many varied candidate answers (multiple robust solvers x diverse starts),
validates each against |f(x)|<=1e-8, and keeps the valid root with the fewest calls,
all under an explicit wall-clock deadline."""
import math
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq

# Safety margin: evaluator kills at 150s; we stop searching well before that.
_DEADLINE = 130.0
_START = time.monotonic()
_MAX_CALLS = 430  # hard cap on f-calls per equation (well under 500)
_TOL = 1e-10      # strict margin, well below the 1e-8 acceptance


def _make_counter(f):
    st = {"n": 0, "over": False}
    def g(x):
        if st["over"]:
            return 0.0
        st["n"] += 1
        if st["n"] > _MAX_CALLS:
            st["over"] = True
            return 0.0
        return f(x)
    return g, st


def _is_bad(x):
    if x is None:
        return True
    if not isinstance(x, (int, float)):
        return True
    if not math.isfinite(x):
        return True
    return False


def _valid(x, g, st):
    """Return True if x is a valid root (uses one extra f-call for verification)."""
    if _is_bad(x) or st["over"]:
        return False
    fx = g(x)  # verification call (counts toward budget)
    if not math.isfinite(fx):
        return False
    return abs(fx) <= _TOL


def _bisect(g, a, b, iters=110):
    try:
        fa = g(a)
        if not math.isfinite(fa):
            return None
        fb = g(b)
        if not math.isfinite(fb):
            return None
        if fa * fb > 0.0:
            return None
        for _ in range(iters):
            m = 0.5 * (a + b)
            fm = g(m)
            if not math.isfinite(fm):
                return None
            if abs(fm) < 1e-15:
                return m
            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)
    except Exception:
        return None


def _secant(g, x0, x1, iters=60):
    try:
        x, px = float(x1), float(x0)
        fpx = g(px)
        if not math.isfinite(fpx):
            return None
        for _ in range(iters):
            fx = g(x)
            if not math.isfinite(fx):
                return None
            if abs(fx) < 1e-15:
                return x
            denom = fx - fpx
            if denom == 0.0:
                return None
            xnew = x - fx * (x - px) / denom
            if not math.isfinite(xnew):
                return None
            px, fpx, x = x, fx, xnew
        return x
    except Exception:
        return None


def _newton_safe(g, x, iters=40, max_step=10.0):
    try:
        x = float(x)
        for _ in range(iters):
            fx = g(x)
            if not math.isfinite(fx):
                return None
            if abs(fx) < 1e-15:
                return x
            h = 1e-5 * max(1.0, abs(x))
            d = (g(x + h) - fx) / h
            if not math.isfinite(d) or d == 0.0:
                return None
            step = fx / d
            s = max(1.0, abs(x))
            if abs(step) > max_step * s:
                step = math.copysign(max_step * s, step)
            xnew = x - step
            if not math.isfinite(xnew):
                return None
            x = xnew
        return x
    except Exception:
        return None


def _scan_brackets(g, x0):
    """Generate a diverse set of (a,b) bracket intervals around x0."""
    x0 = float(x0)
    lo, hi = x0 - 1.0, x0 + 1.0
    brackets = []
    for _ in range(30):
        span = hi - lo
        lo -= span
        hi += span
        f_lo, f_hi = g(lo), g(hi)
        if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi < 0.0:
            brackets.append((lo, hi))
            if len(brackets) >= 3:
                break
    base = max(1e-3, 0.1 * abs(x0))
    for s in (base, base * 0.5, base * 2.0, base * 0.25):
        try:
            if g(x0 - s) * g(x0 + s) < 0.0:
                brackets.append((x0 - s, x0 + s))
        except Exception:
            pass
    return brackets


def _brent(g, a, b):
    try:
        return brentq(g, a, b, xtol=1e-12, rtol=1e-12, maxiter=80)
    except Exception:
        return None


def _newton_polish(g, x, st, tol=1e-14, max_iter=10):
    """High-precision Newton polish to reduce residual further."""
    try:
        x = float(x)
        for _ in range(max_iter):
            if st["over"]:
                break
            fx = g(x)
            if not math.isfinite(fx):
                break
            if abs(fx) < tol:
                break
            h = 1e-6 * max(1.0, abs(x))
            d = (g(x + h) - fx) / h
            if not math.isfinite(d) or abs(d) < 1e-30:
                break
            step = fx / d
            xnew = x - step
            if not math.isfinite(xnew):
                break
            x = xnew
        return x
    except Exception:
        return x


def find_root(f, x0):
    g, st = _make_counter(f)
    best_x, best_n = None, 10 ** 9

    def consider(x):
        nonlocal best_x, best_n
        if x is None or _is_bad(x):
            return
        if st["n"] >= best_n:
            return
        if _valid(x, g, st):
            # Apply Newton polish to get even tighter residual
            polished = _newton_polish(g, x, st)
            if not _is_bad(polished) and st["n"] < best_n:
                # Verify polished result
                fx = g(polished)
                if math.isfinite(fx) and abs(fx) <= _TOL:
                    best_x, best_n = polished, st["n"]
                else:
                    best_x, best_n = x, st["n"]
            else:
                best_x, best_n = x, st["n"]

    x0f = float(x0)

    # (1) Safeguarded Newton from x0 (varied solver, primary starting point)
    consider(_newton_safe(g, x0f))

    # (2) Secant with several varied second points
    for d in (0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 0.2, 2.0):
        consider(_secant(g, x0f, x0f + d))

    # (3) Scan for brackets, then Brent (robust) + bisection on each
    for (a, b) in _scan_brackets(g, x0f):
        consider(_brent(g, a, b))
        consider(_bisect(g, a, b))

    # (4) Fallback: widen window and bisect/Brent
    W = 1.0
    for _ in range(40):
        a, b = x0f - W, x0f + W
        fa, fb = g(a), g(b)
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
            consider(_brent(g, a, b))
            consider(_bisect(g, a, b))
            break
        W *= 2.0

    # Deadline guard: hard stop with a safety margin before the limit.
    if (time.monotonic() - _START) > (_DEADLINE - 5.0):
        if best_x is not None:
            return float(best_x)
        return float(x0f) if not _is_bad(x0f) else 0.0

    if best_x is not None:
        return float(best_x)

    # No validated root: return a finite best-effort point so the rest still score
    # (this equation fails, but we avoid exceptions/None/non-finite).
    return float(x0f) if not _is_bad(x0f) else 0.0
# EVOLVE-BLOCK-END
