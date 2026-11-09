"""Root finder: Brent's method with bracket expansion and a deadline guard."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Hard deadline: evaluator kills at 150 s; finish well before 120 s.
    deadline = time.time() + 110.0
    TOL = 1e-12          # internal tolerance (much tighter than 1e-8)
    XTOL = 1e-15

    x0 = float(x0)

    def _valid(v):
        return math.isfinite(v)

    # Evaluate f once at the start.
    fx0 = f(x0)
    best_x = x0
    best_abs = abs(fx0) if _valid(fx0) else float('inf')

    if _valid(fx0) and best_abs <= TOL:
        return x0

    # ---------- Brent's method (Nelson 1970) ----------
    def brent(a, b, fa, fb, max_iter=200):
        # fa = f(a), fb = f(b), fa*fb < 0 (sign change)
        c = a
        fc = fa
        d = b - a
        e = d
        # We'll track best (x, |f(x)|) seen inside.
        bx = b
        bf = fb
        babs = abs(fb)
        ax = a
        af = fa
        aabs = abs(fa)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if fb * fc > 0:
                c = a
                fc = fa
                d = e = b - a
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
            m = 0.5 * (c - b)
            if abs(m) * abs(fb) <= XTOL * max(abs(b), 1.0) * 0.5:
                break  # x-tolerance met
            if abs(e) * abs(fb) <= TOL:
                break  # f-tolerance met
            if abs(e) >= abs(0.5 * d):
                # Bisection
                s = fb / fc
                e = d
                d = m * s
            else:
                # Interpolation
                A = fc / fb
                B = fa / fb
                C = fa / fc
                if abs(A - 1.0) > 1e-15:
                    A = A - 1.0
                # Inverse quadratic interpolation
                p = C * (2.0 * m * A / (A - B) - (b - c) * C / (A - C))
                q_ = A * (b - a) * B / (B - C) * (c - a) * A / (A - B)
                r = p
                if p > 0.0:
                    r = -p
                elif p < 0.0:
                    r = -p
                # Choose bisection if interpolation is unacceptable
                if (2.0 * r < min(3.0 * m * A - abs(A - 1.0) * q_, abs(e) * A)) \
                   and (2.0 * r < abs(e) * A):
                    e = d
                    d = r / A
                else:
                    e = d
                    d = m
            # Step
            x = b + d
            fx = f(x)
            if not _valid(fx):
                # Non-finite: fall back to bisection step
                x = b + 0.5 * (c - b)
                fx = f(x)
            if not _valid(fx):
                break
            a, fa = b, fb
            b, fb = x, fx
            d = e = c - b
            # Track best
            afx = abs(fx)
            if afx < babs:
                babs = afx
            if afx < aabs:
                aabs = afx
            if abs(fx) < abs(fa):
                # update best among a,b
                pass
        # After loop, best of the two bracket endpoints / last b
        return b, fb

    # ---------- Bracket expansion ----------
    # Try to find a sign change around x0.
    a = x0
    fa = fx0
    if not _valid(fa):
        # x0 not evaluable; pick a small offset
        a = x0 + 1.0
        fa = f(a)

    # Expand geometrically to find b with sign change
    step = 1.0
    b = None
    fb = None
    for _ in range(80):
        if time.time() > deadline:
            break
        # candidate b on both sides
        cand = a + step
        fc = f(cand)
        if _valid(fc) and _valid(fa) and fa * fc < 0:
            b = cand
            fb = fc
            break
        cand2 = a - step
        fc2 = f(cand2)
        if _valid(fc2) and _valid(fa) and fa * fc2 < 0:
            b = cand2
            fb = fc2
            break
        # If fa is not finite, just try to find two points with a sign change
        if not _valid(fa):
            # track best finite pair
            if _valid(fc):
                pass
        step *= 2.0
        if step > 1e12:
            break

    # If we found a bracket, run Brent
    if b is not None and fb is not None and _valid(fa) and _valid(fb) and fa * fb < 0:
        xr, fr = brent(a, b, fa, fb)
        if _valid(fr) and abs(fr) < best_abs:
            best_x = xr
            best_abs = abs(fr)
        if _valid(fr) and abs(fr) <= 1e-8:
            return xr
    elif b is not None and fb is not None and _valid(fb) and abs(fb) < best_abs:
        best_x = b
        best_abs = abs(fb)

    # Fallback: if we still don't have a good root, do a few secant steps from x0
    if best_abs > 1e-8 and _valid(fx0):
        x = x0
        fx = fx0
        # secant with a perturbed second point
        x2 = x0 + 1e-4 * max(1.0, abs(x0))
        try:
            fx2 = f(x2)
        except Exception:
            fx2 = None
        if fx2 is not None and _valid(fx2) and fx2 != fx:
            for _ in range(30):
                if time.time() > deadline:
                    break
                denom = fx2 - fx
                if denom == 0:
                    break
                x_new = x2 - fx2 * (x2 - x) / denom
                if not _valid(x_new):
                    break
                x, fx = x2, fx2
                x2 = x_new
                fx2 = f(x2)
                if not _valid(fx2):
                    break
                if abs(fx2) < best_abs:
                    best_x = x2
                    best_abs = abs(fx2)
                if abs(fx2) <= 1e-8:
                    return x2

    # Final safety: ensure we return a finite float
    if not _valid(best_x):
        best_x = x0 if _valid(x0) else 0.0
    return float(best_x)
# EVOLVE-BLOCK-END
