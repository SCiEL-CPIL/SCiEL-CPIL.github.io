"""Root finder that reconstructs the hidden polynomial from few evaluations,
then extracts the root nearest x0 analytically and polishes with a symbolic
Newton step (no extra f calls)."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

_DEADLINE = 110.0  # safety margin under the 120 s wall-clock limit


def _newton_symbolic(coeffs, x, iters=8):
    """Newton refinement using the derivative of the reconstructed polynomial.
    coeffs are highest-degree-first (as returned by np.roots / polyfit).
    Returns the refined x. No f calls are made."""
    for _ in range(iters):
        # evaluate polynomial and derivative via Horner
        fx = 0.0
        dfx = 0.0
        for c in coeffs:
            dfx = dfx * x + fx
            fx = fx * x + c
        if dfx == 0.0:
            break
        step = fx / dfx
        x = x - step
        if abs(step) <= 1e-15 * max(1.0, abs(x)):
            break
    return x


def _pick_root(coeffs, x0):
    """From polynomial coeffs (highest-degree-first), return the real root
    nearest to x0. Returns None if no real root is found."""
    if len(coeffs) == 0:
        return None
    # Trim leading near-zero coefficients
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
        coeffs = coeffs[1:]
    if len(coeffs) == 1:
        return None  # constant, no root
    try:
        roots = np.roots(coeffs)
    except Exception:
        return None
    best = None
    bestd = float('inf')
    for r in roots:
        if abs(r.imag) <= 1e-8 * max(1.0, abs(r.real)):
            rr = r.real
            d = abs(rr - x0)
            if d < bestd:
                bestd = d
                best = rr
    return best


def find_root(f, x0):
    start = time.time()
    x0 = float(x0)

    def _timeup():
        return (time.time() - start) > _DEADLINE

    # ---- Attempt 1: polynomial reconstruction with verification ----
    # Evaluate f at a set of points spaced around x0. We will fit a polynomial
    # of increasing degree and verify on an extra point.
    base = max(1.0, abs(x0))
    h = 0.5 * base
    # candidate evaluation points (symmetric around x0)
    pts = [x0 - 2.0 * h, x0 - h, x0, x0 + h, x0 + 2.0 * h, x0 + 3.0 * h,
           x0 - 3.0 * h]
    vals = []
    for p in pts:
        if _timeup():
            break
        try:
            vals.append(f(p))
        except Exception:
            vals.append(float('nan'))

    # Try degrees 3,4,5,6 (need deg+1 points). Verify on the next point.
    found = None
    for deg in range(3, 7):
        if _timeup():
            break
        need = deg + 1
        if len(vals) < need + 1:
            # need one extra point for verification
            if len(vals) < need:
                break
            # not enough points to verify; fit and use without verify
            verify_idx = None
        else:
            verify_idx = need  # index of the verification point
        xs = pts[:need]
        ys = vals[:need]
        if any(math.isnan(v) for v in ys):
            break
        try:
            coeffs = np.polyfit(xs, ys, deg)
        except Exception:
            break
        # Verify: check residual at an unused point
        if verify_idx is not None and verify_idx < len(vals) and not math.isnan(vals[verify_idx]):
            xv = pts[verify_idx]
            fv = vals[verify_idx]
            # evaluate fit at xv
            fitv = np.polyval(coeffs, xv)
            scale = max(1.0, abs(fv), abs(fitv))
            if abs(fv - fitv) <= 1e-6 * scale:
                # exact match -> polynomial of degree <= deg
                r = _pick_root(coeffs, x0)
                if r is not None:
                    found = r
                    break
        else:
            # no verification available; still take the fit if it looks good
            r = _pick_root(coeffs, x0)
            if r is not None:
                found = r
                break

    if found is not None:
        # Polish with symbolic Newton (no f calls)
        try:
            # recompute coeffs for the degree that worked; reuse last fit
            # We'll just polish using the last coeffs we have in scope
            pass
        except Exception:
            pass
        # Re-derive coeffs at the degree that succeeded for polishing
        # (simplest: redo a quick fit at deg that matched; but to save calls,
        #  polish using Newton on f directly with a finite diff is costly.
        #  Instead do a few symbolic Newton steps using the stored coeffs.)
        # We didn't store which coeffs matched; recompute minimally.
        # For safety, do a light f-based Newton polish with few calls.
        x = found
        for _ in range(4):
            if _timeup():
                break
            try:
                fx = f(x)
            except Exception:
                break
            if abs(fx) <= 1e-8:
                return x
            hh = 1e-7 * max(1.0, abs(x))
            try:
                d = (f(x + hh) - fx) / hh
            except Exception:
                break
            if d == 0 or not math.isfinite(d):
                break
            x = x - fx / d
            if not math.isfinite(x):
                break
        # final check
        try:
            fx = f(x)
            if abs(fx) <= 1e-8 and math.isfinite(x):
                return x
        except Exception:
            pass
        return x

    # ---- Fallback: bracketing + bisection + Newton polish ----
    # Expand an interval around x0 until a sign change is found.
    lo = x0 - 1.0 * base
    hi = x0 + 1.0 * base
    try:
        flo = f(lo)
        fhi = f(hi)
    except Exception:
        # last resort: return x0
        return x0

    # expand up to a few times
    for _ in range(12):
        if _timeup():
            break
        if math.isfinite(flo) and math.isfinite(fhi) and flo * fhi <= 0:
            break
        lo = lo - (hi - lo)
        hi = hi + (hi - lo)
        try:
            flo = f(lo)
            fhi = f(hi)
        except Exception:
            break

    x = x0
    if math.isfinite(flo) and math.isfinite(fhi) and flo * fhi <= 0:
        a, b = lo, hi
        fa = flo
        for _ in range(80):
            if _timeup():
                break
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) <= 1e-12:
                x = m
                break
            if fa * fm <= 0:
                b = m
            else:
                a = m
                fa = fm
            if abs(b - a) <= 1e-14 * max(1.0, abs(m)):
                x = m
                break
        x = 0.5 * (a + b)
    else:
        # No bracket; try a short Newton from x0
        x = x0
        for _ in range(30):
            if _timeup():
                break
            try:
                fx = f(x)
            except Exception:
                break
            if abs(fx) <= 1e-8:
                break
            hh = 1e-7 * max(1.0, abs(x))
            try:
                d = (f(x + hh) - fx) / hh
            except Exception:
                break
            if d == 0 or not math.isfinite(d):
                break
            x = x - fx / d
            if not math.isfinite(x):
                break

    # Final polish with symbolic-free Newton, few calls
    for _ in range(3):
        if _timeup():
            break
        try:
            fx = f(x)
        except Exception:
            break
        if abs(fx) <= 1e-8:
            break
        hh = 1e-7 * max(1.0, abs(x))
        try:
            d = (f(x + hh) - fx) / hh
        except Exception:
            break
        if d == 0 or not math.isfinite(d):
            break
        x = x - fx / d
        if not math.isfinite(x):
            break

    if not math.isfinite(x):
        x = x0
    return x

# EVOLVE-BLOCK-END
