"""
Root finder: bracket + 1-call/step superlinear driver (secant, IQU-Brent fallback).

Design family preserved: find a sign bracket, then drive a superlinear 1D method
to |f(x)| <= 1e-8, with a cheap final polish. The expensive 3-call/step Newton
derivative is replaced by a 1-call/step secant driver (superlinear order ~1.62),
with an inverse-quadratic-interpolation Brent step as a safeguard. Fewer calls
per step directly lowers mean_calls and improves the score.
"""
import time
import math

TOL = 1e-8
MAX_CALLS = 480


def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0

    x0 = float(x0)
    call_count = 0
    cache = {}

    def _time_ok():
        return (time.time() - start_time) < deadline and call_count < MAX_CALLS

    def _f(x_val):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
        except Exception:
            call_count += 1
            return None
        call_count += 1
        if not math.isfinite(val):
            return None
        cache[key] = val
        return val

    # --- Phase 0: evaluate start point ---
    fx0 = _f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= TOL:
        return x0

    # --- Phase 1: find a bracket [a,b] with a sign change ---
    a = b = fa = fb = None
    step = 0.5
    for _ in range(60):
        if not _time_ok():
            break
        if step > 1e10:
            break
        xr = x0 + step
        fr = _f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx0 * fr < 0:
                a, b, fa, fb = x0, xr, fx0, fr
                break
        xl = x0 - step
        fl = _f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx0 * fl < 0:
                a, b, fa, fb = xl, x0, fl, fx0
                break
        step *= 1.7

    # Wider grid search if no bracket yet
    if a is None or b is None:
        best_x, best_abs = x0, abs(fx0)
        prev_x = prev_f = None
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
            if not _time_ok():
                break
            base = max(abs(x0), 1.0) * scale
            n = 40
            prev_x = prev_f = None
            for i in range(n + 1):
                if not _time_ok():
                    break
                t = i / n
                xt = x0 - base + 2.0 * base * t
                ft = _f(xt)
                if ft is None:
                    prev_x = prev_f = None
                    continue
                if abs(ft) < best_abs:
                    best_abs, best_x = abs(ft), xt
                if abs(ft) <= TOL:
                    return xt
                if prev_f is not None and prev_f * ft < 0:
                    a, b, fa, fb = prev_x, xt, prev_f, ft
                    break
                prev_x, prev_f = xt, ft
            if a is not None:
                break

    # --- Secant driver (1 call/step, superlinear) ---
    def secant_driver(xc, fc, xp, fp, iters):
        for _ in range(iters):
            if not _time_ok():
                break
            if abs(fc) <= TOL:
                return xc
            if abs(fp) <= TOL:
                return xp
            denom = fc - fp
            if abs(denom) < 1e-15:
                break
            xn = xc - fc * (xc - xp) / denom
            if not math.isfinite(xn):
                break
            xp, fp = xc, fc
            xc = xn
            fc = _f(xc)
            if fc is None:
                break
        return xc if (fc is not None and abs(fc) <= abs(fp)) else xp

    # --- IQU-Brent driver (1 call/step, superlinear, bracket-safe) ---
    def brent_driver(a, b, fa, fb, iters):
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            return secant_driver(a, fa, b, fb, iters)
        c, fc = a, fa
        d = e = b - a
        for _ in range(iters):
            if not _time_ok():
                break
            if fb == 0.0:
                return b
            if fc == 0.0:
                return c
            if fa * fb > 0.0:
                a, fa = c, fc
                d = e = b - a
            if abs(e) < 1e-15 or abs(fa) <= abs(fb):
                c, fc = b, fb
                d = e = b - a
            try:
                if abs(d) > 1e-15:
                    s = fa / fb
                    if a != c and c != b:
                        w = fb * (d / (c - b)) * s * (s - fa / fc)
                        t = 1.5 * (c - b) * (fa / fc)
                    else:
                        w, t = 0.0, 0.0
                    if (2.0 * t - w) < (3.0 * t - w) * min(1.0, abs(t) / w) \
                            and (2.0 * t - w) < abs(e):
                        e = d
                        b = c + t
                    else:
                        b = c + (d if d * e > 0.0 else math.copysign(1e-15, d))
                else:
                    e = d = b - a
            except (ZeroDivisionError, OverflowError):
                e = d = b - a
            fb = _f(b)
            if fb is None:
                break
        # best of the three tracked points
        cands = [(a, fa), (b, fb)]
        if fc is not None:
            cands.append((c, fc))
        cands = [p for p in cands if p[1] is not None]
        if not cands:
            return a
        return min(cands, key=lambda p: abs(p[1]))[0]

    # --- If we have a bracket, use the drivers ---
    if a is not None and b is not None and fa is not None and fb is not None:
        if fa * fb < 0.0:
            # Secant driver from the bracket endpoints (1 call/step)
            res = secant_driver(a, fa, b, fb, 60)
            if res is not None and _f(res) is not None and abs(_f(res)) <= TOL:
                return res
            # Fallback: bracket-safe Brent driver
            res = brent_driver(a, b, fa, fb, 60)
            if res is not None and _f(res) is not None and abs(_f(res)) <= TOL:
                return res
            return res if res is not None else a

    # --- No bracket: secant from best point ---
    xc, fc = best_x, _f(best_x)
    if fc is None:
        return best_x
    xp = xc + 1e-4 * max(1.0, abs(xc))
    fp = _f(xp)
    if fp is None:
        return best_x
    res = secant_driver(xc, fc, xp, fp, 70)
    if res is not None:
        fr = _f(res)
        if fr is not None and abs(fr) <= TOL:
            return res
        return res
    return best_x
