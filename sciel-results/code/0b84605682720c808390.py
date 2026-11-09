"""Polynomial-interpolation root finder for 3 fixed smooth polynomial equations."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root(f, x0):
    x0 = float(x0)
    cache = {}
    call_count = [0]
    best_x = x0
    best_abs = None

    def get_f(x):
        x = float(x)
        if not np.isfinite(x):
            return np.inf
        key = x
        if key in cache:
            return cache[key]
        call_count[0] += 1
        try:
            val = float(f(x))
        except Exception:
            val = np.inf
        cache[key] = val
        return val

    def update_best(x, fx):
        nonlocal best_x, best_abs
        if not np.isfinite(x) or not np.isfinite(fx):
            return
        a = abs(fx)
        if best_abs is None or a < best_abs:
            best_abs = a
            best_x = x

    # Initial evaluation at x0
    f0 = get_f(x0)
    update_best(x0, f0)
    if np.isfinite(f0) and abs(f0) <= 1e-8:
        return x0

    # Build initial sample set: x0 and a few nearby points
    points = [(x0, f0)]
    
    # Evaluate 4 more points at strategic locations to build interpolation
    # Use geometric spacing around x0
    for scale in [0.5, 1.0, 2.0, 4.0]:
        for sign in [1.0, -1.0]:
            if call_count[0] >= 9:
                break
            px = x0 + sign * scale
            if not np.isfinite(px):
                continue
            if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
                continue
            fx = get_f(px)
            points.append((px, fx))
            update_best(px, fx)
            if np.isfinite(fx) and abs(fx) <= 1e-8:
                return px
        if call_count[0] >= 9:
            break

    # Main loop: fit polynomial, find roots, evaluate best candidates
    for iteration in range(15):
        if call_count[0] > 45:
            break

        # Get finite points
        finite_pts = [(x, y) for x, y in points if np.isfinite(y)]
        if len(finite_pts) < 3:
            # Expand search if we don't have enough finite points
            scale = max(1.0, 2.0 * max(abs(p[0] - x0) for p in points) if points else 1.0)
            for sign in [1.0, -1.0]:
                px = x0 + sign * scale
                if not np.isfinite(px):
                    continue
                if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
                    continue
                fx = get_f(px)
                points.append((px, fx))
                update_best(px, fx)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return px
            continue

        # Fit polynomial of degree min(len-1, 4)
        xs = np.array([p[0] for p in finite_pts], dtype=float)
        ys = np.array([p[1] for p in finite_pts], dtype=float)
        
        deg = min(len(xs) - 1, 4)
        
        try:
            coeffs = np.polyfit(xs, ys, deg)
            roots = np.roots(coeffs)
        except Exception:
            roots = []

        # Filter roots: real, finite, and reasonably close to our data range
        candidates = []
        if len(xs) > 1:
            x_mean = np.mean(xs)
            x_std = max(1.0, np.std(xs))
            for r in roots:
                if np.isreal(r):
                    rr = float(np.real(r))
                    if np.isfinite(rr) and abs(rr) < 1e10:
                        # Root should be near our data range
                        if abs(rr - x_mean) < 50.0 * x_std:
                            candidates.append(rr)

        # Also add sign-change midpoints from sorted points
        spts = sorted(finite_pts, key=lambda t: t[0])
        for i in range(len(spts) - 1):
            xa, fa = spts[i]
            xb, fb = spts[i + 1]
            if fa * fb < 0:
                if fb != fa:
                    t = fa / (fa - fb)
                    mid = xa + t * (xb - xa)
                else:
                    mid = 0.5 * (xa + xb)
                candidates.append(mid)

        if not candidates:
            # Expand search range
            scale = max(1.0, 2.0 * max(abs(p[0] - x0) for p in points) if points else 1.0)
            for sign in [1.0, -1.0]:
                px = x0 + sign * scale
                if not np.isfinite(px):
                    continue
                if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
                    continue
                fx = get_f(px)
                points.append((px, fx))
                update_best(px, fx)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return px
            continue

        # Sort candidates by proximity to current best
        candidates = sorted(candidates, key=lambda c: abs(c - best_x))
        
        # Evaluate top 2-3 candidates
        evaluated = 0
        for c in candidates[:3]:
            if call_count[0] > 48:
                break
            fc = get_f(c)
            update_best(c, fc)
            if np.isfinite(fc) and abs(fc) <= 1e-8:
                return c
            points.append((c, fc))
            evaluated += 1
            if best_abs is not None and best_abs <= 1e-8:
                return best_x

        if best_abs is not None and best_abs <= 1e-8:
            return best_x

    # Final polish: use the best point and do a few secant steps
    x = best_x
    fx = get_f(x)
    if not np.isfinite(fx):
        return best_x

    # Find second point for secant
    h = 1e-6 * max(1.0, abs(x))
    x2 = x + h
    fx2 = get_f(x2)

    for _ in range(5):
        if call_count[0] > 50:
            break
        if not np.isfinite(fx) or not np.isfinite(fx2) or abs(fx) <= 1e-12:
            break
        if fx2 == fx:
            break
        step = fx * (x - x2) / (fx - fx2)
        if not np.isfinite(step) or abs(step) > 10.0 * max(1.0, abs(x)):
            break
        xn = x - step
        if not np.isfinite(xn):
            break
        fxn = get_f(xn)
        if not np.isfinite(fxn):
            break
        if abs(fxn) < abs(fx):
            x, fx = xn, fxn
            x2, fx2 = x, fx
        else:
            x2, fx2 = xn, fxn
        if abs(fx) <= 1e-12:
            break

    update_best(x, fx)
    return best_x
# EVOLVE-BLOCK-END
