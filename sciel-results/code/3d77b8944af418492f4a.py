"""Root finding for ill-conditioned equations: model-and-optimize approach."""
import time
import math
import numpy as np
from scipy.optimize import minimize

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    time_limit = 145.0  # safety margin: 150s wall-clock, leave 5s

    best_x = float(x0)
    try:
        best_fx = float(f(best_x))
        eval_count += 1
        best_abs_fx = abs(best_fx) if math.isfinite(best_fx) else float('inf')
    except Exception:
        best_fx = 0.0
        best_abs_fx = float('inf')

    def _time_left():
        return time_limit - (time.time() - start_time)

    def _eval(x):
        """Evaluate f(x) with eval-count and time guards. Returns (fx, ok)."""
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

    # ------------------------------------------------------------------
    # Phase 1: Adaptive secant with model-based step control
    # ------------------------------------------------------------------
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

    # Track model quality: if secant steps are making progress, keep going
    prev_abs_f = abs(f2)
    consecutive_bad = 0
    max_consecutive_bad = 8

    for it in range(200):
        if _check_done():
            return best_x
        if eval_count >= max_evals or _time_left() < 2.0:
            break

        dx = x2 - x1
        if abs(dx) < 1e-15:
            break
        d = (f2 - f1) / dx
        if abs(d) < 1e-15:
            break

        # Model-based: compute secant step
        x_new = x2 - f2 / d

        # Adaptive safeguard: limit step based on how well the model fits
        # If we're making good progress, allow larger steps
        max_step = 10.0 * max(1.0, abs(x2))
        if abs(x_new - x2) > max_step:
            x_new = x2 - math.copysign(max_step, x_new - x2)

        f_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        # Track progress
        if abs(f_new) < abs(f2) * 0.95:
            consecutive_bad = 0
        else:
            consecutive_bad += 1

        # If too many bad steps in a row, break out
        if consecutive_bad >= max_consecutive_bad:
            break

        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Bracket search + bisection with inverse quadratic interpolation
    # ------------------------------------------------------------------
    a = x2
    fa = f2
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    # Expand bracket outward
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
        scale *= 1.5

    if b is not None and fb is not None and fa * fb < 0:
        lo, hi = min(a, b), max(a, b)
        flo = fa if a == lo else fb
        fhi = fa if a == hi else fb
        if flo * fhi > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        for _ in range(80):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break

            if abs(flo) <= tol:
                return lo
            if abs(fhi) <= tol:
                return hi

            # Inverse quadratic interpolation
            s = None
            denom = fhi * (flo + fhi) + flo * (fhi - flo)
            if denom != 0:
                s = lo + (hi - lo) * flo * (flo - fhi) / denom
                if not math.isfinite(s) or s <= lo or s >= hi:
                    s = None

            if s is None:
                s = 0.5 * (lo + hi)

            fs, ok = _eval(s)
            if not ok:
                break

            if abs(fs) <= tol:
                return s

            if flo * fs < 0:
                hi, fhi = s, fs
            else:
                lo, flo = s, fs

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3: Nelder-Mead optimization on g(x) = f(x)^2
    # ------------------------------------------------------------------
    def _g(x):
        """Surrogate objective: f(x)^2. Returns (g_val, ok)."""
        fx, ok = _eval(x)
        if not ok:
            return None, False
        return fx * fx, True

    # Build initial simplex around best_x
    scale_nm = 1e-3 * max(1.0, abs(best_x))
    simplex = [best_x, best_x + scale_nm, best_x - scale_nm]
    gvals = []
    valid_simplex = True
    for xi in simplex:
        gi, ok = _g(xi)
        if not ok:
            valid_simplex = False
            break
        gvals.append(gi)
        if gi <= tol * tol:
            return xi

    if valid_simplex:
        alpha = 1.5
        gamma = 2.5
        rho = 0.4
        sigma = 0.4
        max_nm_iter = 80

        for nm_it in range(max_nm_iter):
            if _check_done():
                return best_x
            if eval_count >= max_evals - 5 or _time_left() < 1.0:
                break

            # Sort simplex by g value
            order = sorted(range(3), key=lambda i: gvals[i])
            simplex = [simplex[i] for i in order]
            gvals = [gvals[i] for i in order]

            # Check convergence
            if gvals[0] <= tol * tol:
                return simplex[0]
            if abs(gvals[2] - gvals[0]) < 1e-16:
                break

            # Centroid of all but worst
            centroid = (simplex[0] + simplex[1]) / 2.0

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[2])
            g_ref, ok = _g(x_ref)
            if not ok:
                break

            if g_ref <= gvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                g_exp, ok2 = _g(x_exp)
                if ok2:
                    if g_exp < g_ref:
                        simplex[2] = x_exp
                        gvals[2] = g_exp
                    else:
                        simplex[2] = x_ref
                        gvals[2] = g_ref
                else:
                    simplex[2] = x_ref
                    gvals[2] = g_ref
            elif g_ref < gvals[1]:
                simplex[2] = x_ref
                gvals[2] = g_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[2] - centroid)
                g_con, ok3 = _g(x_con)
                if ok3 and g_con < gvals[2]:
                    simplex[2] = x_con
                    gvals[2] = g_con
                else:
                    # Shrink
                    for i in range(2):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        g_i, ok4 = _g(simplex[i])
                        if not ok4:
                            break
                        gvals[i] = g_i
                    else:
                        continue
                    break

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3b: L-BFGS-B optimization on g(x) = f(x)^2
    # ------------------------------------------------------------------
    if eval_count < max_evals - 20 and _time_left() > 5.0:
        try:
            def _objective(x_vec):
                """Objective for L-BFGS-B: f(x)^2 with eval counting."""
                nonlocal eval_count, best_x, best_fx, best_abs_fx
                if eval_count >= max_evals or _time_left() <= 1.0:
                    return float('inf')
                x_val = float(x_vec[0])
                try:
                    fx = float(f(x_val))
                except Exception:
                    return float('inf')
                eval_count += 1
                if not math.isfinite(fx):
                    return float('inf')
                abs_fx = abs(fx)
                if abs_fx < best_abs_fx:
                    best_x = x_val
                    best_fx = fx
                    best_abs_fx = abs_fx
                return fx * fx
            
            # Initial guess: use best_x found so far
            x_start = np.array([best_x])
            
            # Run L-BFGS-B with tight tolerances
            result = minimize(
                _objective, 
                x_start, 
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-12,
                    'ftol': 1e-12,
                    'maxls': 20
                }
            )
            
            if result.success and result.fun <= tol * tol:
                return float(result.x[0])
                
            # Even if not fully converged, check if we found a better point
            if result.fun < best_abs_fx * best_abs_fx:
                best_x = float(result.x[0])
                best_fx = float(f(best_x)) if eval_count < max_evals else best_fx
                if eval_count < max_evals and math.isfinite(best_fx):
                    eval_count += 1
                    best_abs_fx = abs(best_fx)
                    if best_abs_fx <= tol:
                        return best_x
                        
        except Exception:
            pass

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3c: Multi-start secant from diverse points
    # ------------------------------------------------------------------
    # Adaptive offsets: use local scale from recent iterations
    local_scale = max(1.0, abs(best_x))
    # Estimate local step size from the secant history
    if abs(x2 - x1) > 1e-15:
        local_step = abs(x2 - x1)
        if local_step < 1e-10:
            local_step = 1e-4 * local_scale
    else:
        local_step = 1e-4 * local_scale
    
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
               0.01, -0.01, 0.5, -0.5, 5.0, -5.0,
               local_step, -local_step, 2*local_step, -2*local_step,
               0.5*local_step, -0.5*local_step]
    # Deduplicate and filter
    seen = set()
    unique_offsets = []
    for off in offsets:
        key = round(off, 12)
        if key not in seen and abs(off) > 1e-15:
            seen.add(key)
            unique_offsets.append(off)
    for off in unique_offsets:
        if eval_count >= max_evals - 20 or _time_left() < 2.0:
            break
        x_try = float(x0) + off * max(1.0, abs(x0))
        fx_try, ok = _eval(x_try)
        if not ok:
            continue
        if abs(fx_try) <= tol:
            return x_try

        # Quick secant from this point
        h_r = 1e-4 * max(1.0, abs(x_try))
        x_r2 = x_try + h_r
        fx_r2, ok2 = _eval(x_r2)
        if not ok2:
            x_r2 = x_try - h_r
            fx_r2, ok2 = _eval(x_r2)
        if not ok2:
            continue

        consecutive_bad_r = 0
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
            x_new = x_r2 - fx_r2 / d_r
            fx_new, ok3 = _eval(x_new)
            if not ok3:
                break
            if abs(fx_new) <= tol:
                return x_new
            if abs(fx_new) < abs(fx_r2) * 0.95:
                consecutive_bad_r = 0
            else:
                consecutive_bad_r += 1
                if consecutive_bad_r >= 4:
                    break
            x_try, fx_try = x_r2, fx_r2
            x_r2, fx_r2 = x_new, fx_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 4: High-precision polish pass with tighter tolerance
    # ------------------------------------------------------------------
    if eval_count < max_evals - 10 and _time_left() > 2.0:
        if abs(best_fx) < 1e-2:
            # Try 3-point inverse quadratic interpolation
            x_polish = best_x
            fx_polish = best_fx
            # Get two nearby points for IQU
            h_p = 1e-6 * max(1.0, abs(x_polish))
            x_a = x_polish - h_p
            x_b = x_polish + h_p
            fa_p, ok_a = _eval(x_a)
            fb_p, ok_b = _eval(x_b)
            
            if ok_a and ok_b and abs(fa_p) < 1.0 and abs(fb_p) < 1.0:
                # We have three points: (x_a, fa_p), (x_polish, fx_polish), (x_b, fb_p)
                # Use inverse quadratic interpolation (IQU)
                for _ in range(60):
                    if _check_done():
                        return best_x
                    if eval_count >= max_evals or _time_left() < 1.0:
                        break
                    
                    xa, ya = x_a, fa_p
                    xb, yb = x_polish, fx_polish
                    xc, yc = x_b, fb_p
                    
                    # IQU formula: x_new = (xa^2*yb*yc*(yb-yc) + xb^2*ya*yc*(yc-ya) + xc^2*ya*yb*(ya-yb))
                    #                    / (ya*yb*(ya-yb) + yb*yc*(yb-yc) + yc*ya*(yc-ya))
                    denom2 = ya*yb*(ya-yb) + yb*yc*(yb-yc) + yc*ya*(yc-ya)
                    if abs(denom2) < 1e-30:
                        break
                    
                    num2 = xa*xa*yb*yc*(yb-yc) + xb*xb*ya*yc*(yc-ya) + xc*xc*ya*yb*(ya-yb)
                    x_new = num2 / denom2
                    
                    if not math.isfinite(x_new):
                        break
                    
                    fx_new, ok = _eval(x_new)
                    if not ok:
                        break
                    if abs(fx_new) <= tol:
                        return x_new
                    
                    # Replace the worst point (largest |f|)
                    if abs(fx_new) < abs(fa_p):
                        x_a, fa_p = x_new, fx_new
                    elif abs(fx_new) < abs(fx_polish):
                        x_polish, fx_polish = x_new, fx_new
                    else:
                        x_b, fb_p = x_new, fx_new
            
            # Fallback: focused secant polish with adaptive step
            if not _check_done():
                x_s = best_x
                fx_s = best_fx
                # Use the best derivative estimate from recent history
                if abs(x2 - x1) > 1e-15:
                    d_est = (f2 - f1) / (x2 - x1)
                else:
                    d_est = None
                
                for _ in range(80):
                    if _check_done():
                        return best_x
                    if eval_count >= max_evals or _time_left() < 0.5:
                        break
                    
                    if d_est is not None and abs(d_est) > 1e-15:
                        # Newton step using estimated derivative
                        x_new = x_s - fx_s / d_est
                    else:
                        # Secant with tiny perturbation
                        h_n = 1e-10 * max(1.0, abs(x_s))
                        x_new = x_s + h_n
                        fx_new, ok = _eval(x_new)
                        if not ok:
                            break
                        dx_n = x_new - x_s
                        if abs(dx_n) < 1e-15:
                            break
                        d_n = (fx_new - fx_s) / dx_n
                        if abs(d_n) < 1e-15:
                            break
                        x_new = x_s - fx_s / d_n
                        d_est = d_n
                    
                    if not math.isfinite(x_new):
                        break
                    
                    fx_new, ok = _eval(x_new)
                    if not ok:
                        break
                    if abs(fx_new) <= tol:
                        return x_new
                    
                    # Update derivative estimate
                    dx_upd = x_new - x_s
                    if abs(dx_upd) > 1e-15:
                        d_est = (fx_new - fx_s) / dx_upd
                    
                    if abs(fx_new) < abs(fx_s) * 0.999:
                        x_s, fx_s = x_new, fx_new
                    else:
                        break

    return best_x
