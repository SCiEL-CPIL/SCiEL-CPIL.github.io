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

    # Build initial sample set: x0 and 2 nearby points (minimal for cubic)
    points = [(x0, f0)]
    
    # Evaluate 2 more points at strategic locations
    for scale in [1.0, -1.0]:
        if call_count[0] >= 3:
            break
        px = x0 + scale
        if not np.isfinite(px):
            continue
        if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
            continue
        fx = get_f(px)
        points.append((px, fx))
        update_best(px, fx)
        if np.isfinite(fx) and abs(fx) <= 1e-8:
            return px

    # Main loop: fit polynomial, find roots, evaluate best candidates
    for iteration in range(20):
        if call_count[0] > 40:
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

        # Add inverse quadratic interpolation (IQI) candidate if we have 3+ points
        if len(finite_pts) >= 3:
            try:
                # Use the 3 points with smallest |f| values for IQI
                sorted_by_f = sorted(finite_pts, key=lambda t: abs(t[1]))
                x1, f1 = sorted_by_f[0]
                x2, f2 = sorted_by_f[1]
                x3, f3 = sorted_by_f[2]
                
                # Inverse quadratic interpolation formula
                d1 = (x1 - x2) * (x1 - x3)
                d2 = (x2 - x1) * (x2 - x3)
                d3 = (x3 - x1) * (x3 - x2)
                
                if abs(d1) > 1e-15 and abs(d2) > 1e-15 and abs(d3) > 1e-15:
                    iq_x = (f1 * f2 * f3) / (
                        f1 * f2 / d1 + f1 * f3 / d2 + f2 * f3 / d3
                    ) if (f1 * f2 / d1 + f1 * f3 / d2 + f2 * f3 / d3) != 0 else x1
                    if np.isfinite(iq_x) and abs(iq_x) < 1e10:
                        candidates.append(iq_x)
            except Exception:
                pass

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

        # Deduplicate candidates
        unique_candidates = []
        for c in candidates:
            if not any(abs(c - uc) <= 1e-12 * max(1.0, abs(c), abs(uc)) for uc in unique_candidates):
                unique_candidates.append(c)
        candidates = unique_candidates

        # Sort candidates by proximity to current best
        candidates = sorted(candidates, key=lambda c: abs(c - best_x))
        
        # Evaluate top 1-2 candidates (be conservative on calls)
        evaluated = 0
        for c in candidates[:2]:
            if call_count[0] > 45:
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

        # If we've made good progress, try to refine with secant
        if best_abs is not None and best_abs < 1e-4 and call_count[0] < 45:
            break

    # Final polish: use the best point and do secant/Newton refinement
    x = best_x
    fx = get_f(x)
    if not np.isfinite(fx):
        return best_x

    # Find second point for secant - use the point with second-best |f|
    second_x = None
    second_fx = None
    for px, pf in points:
        if px != x and np.isfinite(pf):
            if second_fx is None or abs(pf) < abs(second_fx):
                second_x = px
                second_fx = pf
    
    if second_x is None:
        h = 1e-6 * max(1.0, abs(x))
        second_x = x + h
        second_fx = get_f(second_x)

    # Secant refinement
    for _ in range(8):
        if call_count[0] > 48:
            break
        if not np.isfinite(fx) or not np.isfinite(second_fx) or abs(fx) <= 1e-12:
            break
        if second_fx == fx:
            break
        step = fx * (x - second_x) / (fx - second_fx)
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
            second_x, second_fx = x, fx
        else:
            second_x, second_fx = xn, fxn
        if abs(fx) <= 1e-12:
            break

    update_best(x, fx)
    return best_x
# EVOLVE-BLOCK-END
