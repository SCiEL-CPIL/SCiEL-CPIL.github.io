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
    deadline = 145.0

    # State to track the best point seen so far
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
        # Update best
        abs_fx = abs(fx)
        if abs_fx < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs_fx
        return fx, True

    def _check_done():
        return best_abs_fx <= tol

    # If already good enough
    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 1: Secant/Newton hybrid (1-2 evals per iteration)
    # ------------------------------------------------------------------
    x1 = float(x0)
    f1 = best_fx
    # Get second point for secant
    h_init = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h_init
    f2, ok = _eval(x2)
    if not ok:
        # Try negative direction
        x2 = x1 - h_init
        f2, ok = _eval(x2)
    if not ok:
        return best_x

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

        # Use Newton step if we're close enough (|f| < 1e-6) for faster quadratic convergence
        if abs(f2) < 1e-6 and eval_count < max_evals - 5:
            # Finite difference derivative
            h_fd = max(1e-8, 1e-8 * abs(x2))
            f2_plus, ok_fd = _eval(x2 + h_fd)
            if ok_fd:
                d_fd = (f2_plus - f2) / h_fd
                if abs(d_fd) > 1e-15:
                    x_new = x2 - f2 / d_fd
                else:
                    x_new = x2 - f2 / d
            else:
                x_new = x2 - f2 / d
        else:
            x_new = x2 - f2 / d

        # Safeguard: limit step size
        max_step = 10.0 * max(1.0, abs(x2))
        if abs(x_new - x2) > max_step:
            x_new = x2 - math.copysign(max_step, x_new - x2)

        f_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        # Check for stagnation
        if abs(f_new) >= abs(f2) * 1.1 and abs(x_new - x2) < 1e-10:
            break

        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Bracket search + manual Brent-like method
    # ------------------------------------------------------------------
    a = x2
    fa = f2
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    # Expand bracket outward in both directions
    b = None
    fb = None
    scale = 1.0
    for _ in range(25):
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
        # Manual inverse quadratic interpolation + bisection (Brent-like)
        # Use our _eval to stay within budget
        lo, hi = min(a, b), max(a, b)
        flo, fhi = fa if a == lo else fb, fa if a == hi else fb
        # Ensure flo*fhi < 0
        if flo * fhi > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        c = lo
        fc = flo
        d = hi - lo
        e_val = d

        for _ in range(80):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break

            # Check if root is at one of the endpoints
            if abs(flo) <= tol:
                return lo
            if abs(fhi) <= tol:
                return hi

            # Inverse quadratic interpolation (Brent's IQR)
            s = None
            denom = (fhi - flo)
            if abs(denom) > 1e-15:
                # Standard IQR: s = hi - fhi * (hi - lo) / (fhi - flo)
                # This is the secant-like step from the bracket
                s = hi - fhi * (hi - lo) / (fhi - flo)
                # Alternatively, use the more stable form:
                # s = lo - flo * (hi - lo) / (fhi - flo)
                # Use the one that stays closer to the root
                s_alt = lo - flo * (hi - lo) / (fhi - flo)
                # Pick the one that's within [lo, hi] or closer to midpoint
                mid = 0.5 * (lo + hi)
                if lo < s < hi:
                    pass  # s is good
                elif lo < s_alt < hi:
                    s = s_alt
                else:
                    s = mid
            else:
                s = 0.5 * (lo + hi)

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

    # ------------------------------------------------------------------
    # Phase 3: Continued secant from best point
    # ------------------------------------------------------------------
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
        x_new = x2 - f2 / d

        max_step = 5.0 * max(1.0, abs(x2))
        if abs(x_new - x2) > max_step:
            x_new = x2 - math.copysign(max_step, x_new - x2)

        f_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        if abs(f_new) >= abs(f2) * 1.1 and abs(x_new - x2) < 1e-10:
            break

        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 4: Restart portfolio with secant/Newton hybrid
    # ------------------------------------------------------------------
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]
    for off in offsets:
        if eval_count >= max_evals - 20 or _time_left() < 3.0:
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
            continue

        for _ in range(20):
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
            # Use Newton if close
            if abs(fx_r2) < 1e-6 and eval_count < max_evals - 5:
                h_fd_r = max(1e-8, 1e-8 * abs(x_r2))
                f_r2_plus, ok_fd = _eval(x_r2 + h_fd_r)
                if ok_fd:
                    d_fd_r = (f_r2_plus - fx_r2) / h_fd_r
                    if abs(d_fd_r) > 1e-15:
                        x_new = x_r2 - fx_r2 / d_fd_r
                    else:
                        x_new = x_r2 - fx_r2 / d_r
                else:
                    x_new = x_r2 - fx_r2 / d_r
            else:
                x_new = x_r2 - fx_r2 / d_r
            fx_new, ok3 = _eval(x_new)
            if not ok3:
                break
            if abs(fx_new) <= tol:
                return x_new
            if abs(fx_new) >= abs(fx_r2) * 1.1 and abs(x_new - x_r2) < 1e-10:
                break
            x_try, fx_try = x_r2, fx_r2
            x_r2, fx_r2 = x_new, fx_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 5: Final high-precision polish pass (Newton with FD derivative)
    # ------------------------------------------------------------------
    if eval_count < max_evals - 15 and _time_left() > 3.0:
        x_polish = best_x
        fx_polish = best_fx
        if abs(fx_polish) < 1e-4:  # Only polish if reasonably close
            polish_iters = 0
            max_polish = 30
            while polish_iters < max_polish and eval_count < max_evals - 5 and _time_left() > 1.0:
                if _check_done():
                    return best_x
                # Newton step with central finite difference derivative
                h_fd = max(1e-9, 1e-9 * abs(x_polish))
                f_plus, ok1 = _eval(x_polish + h_fd)
                if not ok1:
                    break
                f_minus, ok2 = _eval(x_polish - h_fd)
                if not ok2:
                    # Fall back to forward difference
                    d_fd = (f_plus - fx_polish) / h_fd
                else:
                    d_fd = (f_plus - f_minus) / (2.0 * h_fd)
                if abs(d_fd) < 1e-15:
                    break
                x_new = x_polish - fx_polish / d_fd
                fx_new, ok = _eval(x_new)
                if not ok:
                    break
                if abs(fx_new) <= tol:
                    return x_new
                if abs(fx_new) < abs(fx_polish) * 0.5:
                    x_polish, fx_polish = x_new, fx_new
                elif abs(fx_new) < abs(fx_polish):
                    # Accept but with damping
                    x_polish = 0.5 * (x_polish + x_new)
                    fx_polish, _ = _eval(x_polish)
                    if fx_polish is None:
                        break
                else:
                    break
                polish_iters += 1

    return best_x
