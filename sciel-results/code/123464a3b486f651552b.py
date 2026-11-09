"""Root finding for ill-conditioned equations: robust adaptive hybrid with polish."""
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    deadline = 142.0

    best_x = float(x0)
    try:
        best_fx = float(f(best_x))
        eval_count += 1
        best_abs_fx = abs(best_fx) if math.isfinite(best_fx) else float('inf')
    except Exception:
        best_fx = 0.0
        best_abs_fx = float('inf')

    def _time_left():
        return deadline - (time.time() - start_time)

    def _eval(x):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals or _time_left() <= 0.5:
            return None, False
        try:
            fx = float(f(x))
        except Exception:
            return None, False
        eval_count += 1
        if not math.isfinite(fx):
            return None, False
        abs_fx = abs(fx)
        if abs_fx < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs_fx
        return fx, True

    def _check_done():
        return best_abs_fx <= tol

    if _check_done():
        return best_x

    # Phase 1: Secant with adaptive damping
    x1 = float(x0)
    f1 = best_fx
    h_init = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h_init
    f2, ok = _eval(x2)
    if not ok:
        x2 = x1 - h_init
        f2, ok = _eval(x2)
    if not ok:
        return best_x

    # Track secant history for detecting stagnation
    prev_abs_fx = abs(f2)
    stagnant = 0

    secant_iters = 0
    max_secant = 200
    while secant_iters < max_secant and eval_count < max_evals and _time_left() > 2.0:
        if _check_done():
            return best_x
        dx = x2 - x1
        if abs(dx) < 1e-15:
            break
        d = (f2 - f1) / dx
        if abs(d) < 1e-15:
            break

        # Compute secant step
        step = f2 / d
        x_new = x2 - step

        # Damping for ill-conditioned problems
        max_step = 10.0 * max(1.0, abs(x2))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            x_new = x2 - step

        f_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        # Check for progress
        if abs(f_new) < abs(f2) * 0.95:
            stagnant = 0
        else:
            stagnant += 1
            if stagnant > 5:
                break

        prev_abs_fx = abs(f2)
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # Phase 2: Bracket search + Brent-like method
    a = x2
    fa = f2
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    b = None
    fb = None
    scale = 1.0
    for _ in range(30):
        if _time_left() < 2.0 or eval_count >= max_evals - 15:
            break
        b_pos = a + scale * max(1.0, abs(a))
        fb_pos, ok1 = _eval(b_pos)
        if ok1 and fa * fb_pos < 0:
            b = b_pos
            fb = fb_pos
            break
        b_neg = a - scale * max(1.0, abs(a))
        fb_neg, ok2 = _eval(b_neg)
        if ok2 and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            break
        scale *= 1.7

    if b is not None and fb is not None and fa * fb < 0:
        lo, hi = min(a, b), max(a, b)
        flo, fhi = fa if a == lo else fb, fa if a == hi else fb
        if flo * fhi > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        c = lo
        fc = flo
        d_val = hi - lo
        e_val = d_val

        for _ in range(80):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break

            if abs(flo) <= tol:
                return lo
            if abs(fhi) <= tol:
                return hi

            # Inverse quadratic interpolation with fallback
            s = None
            denom = fhi * (flo + fhi) + flo * (fhi - flo)
            if denom != 0:
                s = lo + (hi - lo) * flo * (flo - fhi) / denom
            if s is None or not math.isfinite(s) or s <= lo or s >= hi:
                s = 0.5 * (lo + hi)

            fs, ok = _eval(s)
            if not ok:
                break

            if abs(fs) <= tol:
                return s

            if flo * fs < 0:
                hi = s
                fhi = fs
            else:
                lo = s
                flo = fs

    if _check_done():
        return best_x

    # Phase 2.5: Nelder-Mead on g(x) = f(x)^2 (derivative-free global search)
    if eval_count < max_evals - 30 and _time_left() > 5.0:
        # Initialize simplex around best_x
        scale_nm = max(1e-3, 1e-4 * abs(best_x))
        simplex_x = [best_x, best_x + scale_nm, best_x - scale_nm]
        simplex_g = []
        simplex_valid = True
        for xi in simplex_x:
            gxi, ok = _eval(xi)
            if not ok:
                simplex_valid = False
                break
            simplex_g.append(gxi * gxi)  # g(x) = f(x)^2

        if simplex_valid:
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            max_nm_iters = 60
            for it in range(max_nm_iters):
                if _check_done():
                    return best_x
                if eval_count >= max_evals or _time_left() < 2.0:
                    break

                # Sort by g value
                order = sorted(range(3), key=lambda i: simplex_g[i])
                simplex_x = [simplex_x[i] for i in order]
                simplex_g = [simplex_g[i] for i in order]

                # Centroid of best 2 points
                cx = 0.5 * (simplex_x[0] + simplex_x[1])

                # Reflection
                xr = cx + alpha * (cx - simplex_x[2])
                gr, ok = _eval(xr)
                if not ok:
                    break
                gr = gr * gr

                if gr < simplex_g[0]:
                    # Try expansion
                    xe = cx + gamma * (cx - simplex_x[2])
                    ge, ok = _eval(xe)
                    if not ok:
                        ge = gr
                        xe = xr
                    if ge < gr:
                        simplex_x[2] = xe
                        simplex_g[2] = ge
                    else:
                        simplex_x[2] = xr
                        simplex_g[2] = gr
                elif gr < simplex_g[1]:
                    # Accept reflection
                    simplex_x[2] = xr
                    simplex_g[2] = gr
                else:
                    # Contraction
                    if gr < simplex_g[2]:
                        # Outside contraction
                        xc = cx + rho * (xr - cx)
                    else:
                        # Inside contraction
                        xc = cx - rho * (xr - cx)
                    gc, ok = _eval(xc)
                    if not ok:
                        break
                    gc = gc * gc
                    if gc <= min(gr, simplex_g[2]):
                        simplex_x[2] = xc
                        simplex_g[2] = gc
                    else:
                        # Shrink
                        for i in range(1, 3):
                            simplex_x[i] = simplex_x[0] + sigma * (simplex_x[i] - simplex_x[0])
                            gi, ok = _eval(simplex_x[i])
                            if not ok:
                                break
                            simplex_g[i] = gi * gi
                        else:
                            continue
                        break

                # Check convergence
                if max(simplex_g) - min(simplex_g) < 1e-16:
                    break

        # After Nelder-Mead, try secant from best NM point
        best_nm_x = simplex_x[0] if simplex_valid else best_x
        if eval_count < max_evals - 10 and _time_left() > 3.0:
            x1 = best_nm_x
            f1, ok = _eval(x1)
            if ok:
                h_nm = 1e-6 * max(1.0, abs(x1))
                x2 = x1 + h_nm
                f2, ok = _eval(x2)
                if not ok:
                    x2 = x1 - h_nm
                    f2, ok = _eval(x2)
                if ok:
                    stagnant = 0
                    for _ in range(30):
                        if _check_done():
                            return best_x
                        if eval_count >= max_evals or _time_left() < 1.0:
                            break
                        dx = x2 - x1
                        if abs(dx) < 1e-15:
                            break
                        d = (f2 - f1) / dx
                        if abs(d) < 1e-15:
                            break
                        step = f2 / d
                        x_new = x2 - step
                        max_step = 5.0 * max(1.0, abs(x2))
                        if abs(step) > max_step:
                            step = math.copysign(max_step, step)
                            x_new = x2 - step
                        f_new, ok = _eval(x_new)
                        if not ok:
                            break
                        if abs(f_new) <= tol:
                            return x_new
                        if abs(f_new) < abs(f2) * 0.95:
                            stagnant = 0
                        else:
                            stagnant += 1
                            if stagnant > 5:
                                break
                        x1, f1 = x2, f2
                        x2, f2 = x_new, f_new

    if _check_done():
        return best_x

    # Phase 3: Secant restart from best point
    x1 = best_x
    f1 = best_fx
    h = 1e-6 * max(1.0, abs(x1))
    x2 = x1 + h
    f2, ok = _eval(x2)
    if not ok:
        x2 = x1 - h
        f2, ok = _eval(x2)
    if not ok:
        return best_x

    stagnant = 0
    secant_iters = 0
    max_secant2 = 100
    while secant_iters < max_secant2 and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x
        dx = x2 - x1
        if abs(dx) < 1e-15:
            break
        d = (f2 - f1) / dx
        if abs(d) < 1e-15:
            break

        step = f2 / d
        x_new = x2 - step

        max_step = 5.0 * max(1.0, abs(x2))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            x_new = x2 - step

        f_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        if abs(f_new) < abs(f2) * 0.95:
            stagnant = 0
        else:
            stagnant += 1
            if stagnant > 5:
                break

        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # Phase 4: Multi-scale restart portfolio
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0]
    for off in offsets:
        if eval_count >= max_evals - 15 or _time_left() < 2.0:
            break
        x_try = float(x0) + off * max(1.0, abs(x0))
        fx_try, ok = _eval(x_try)
        if not ok:
            continue
        if abs(fx_try) <= tol:
            return x_try

        h_r = 1e-4 * max(1.0, abs(x_try))
        x_r2 = x_try + h_r
        fx_r2, ok2 = _eval(x_r2)
        if not ok2:
            continue

        stagnant = 0
        for _ in range(15):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            dx_r = x_r2 - x_try
            if abs(dx_r) < 1e-15:
                break
            d_r = (fx_r2 - fx_try) / dx_r
            if abs(d_r) < 1e-15:
                break

            step_r = fx_r2 / d_r
            x_new = x_r2 - step_r

            max_step_r = 3.0 * max(1.0, abs(x_r2))
            if abs(step_r) > max_step_r:
                step_r = math.copysign(max_step_r, step_r)
                x_new = x_r2 - step_r

            fx_new, ok3 = _eval(x_new)
            if not ok3:
                break

            if abs(fx_new) <= tol:
                return x_new

            if abs(fx_new) < abs(fx_r2) * 0.95:
                stagnant = 0
            else:
                stagnant += 1
                if stagnant > 3:
                    break

            x_try, fx_try = x_r2, fx_r2
            x_r2, fx_r2 = x_new, fx_new

    if _check_done():
        return best_x

    # Phase 5: Final high-precision polish
    if eval_count < max_evals - 10 and _time_left() > 2.0:
        x_polish = best_x
        fx_polish = best_fx
        if abs(fx_polish) < 1e-5:
            h_p = 1e-8 * max(1.0, abs(x_polish))
            xp2 = x_polish + h_p
            fp2, ok = _eval(xp2)
            if ok:
                stagnant = 0
                for _ in range(40):
                    if _check_done():
                        return best_x
                    if eval_count >= max_evals or _time_left() < 1.0:
                        break
                    dx_p = xp2 - x_polish
                    if abs(dx_p) < 1e-15:
                        break
                    d_p = (fp2 - fx_polish) / dx_p
                    if abs(d_p) < 1e-15:
                        break

                    step_p = fp2 / d_p
                    x_new = xp2 - step_p

                    max_step_p = 2.0 * max(1.0, abs(xp2))
                    if abs(step_p) > max_step_p:
                        step_p = math.copysign(max_step_p, step_p)
                        x_new = xp2 - step_p

                    fx_new, ok = _eval(x_new)
                    if not ok:
                        break

                    if abs(fx_new) <= tol:
                        return x_new

                    if abs(fx_new) < abs(fp2) * 0.99:
                        stagnant = 0
                        x_polish, fx_polish = xp2, fp2
                        xp2, fp2 = x_new, fx_new
                    else:
                        stagnant += 1
                        if stagnant > 3:
                            break

    return best_x
