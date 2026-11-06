"""Root finding for ill-conditioned equations: safeguarded global search + high-precision polish."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Wall-clock deadline guard: return best result found with a safety margin.
    deadline = time.time() + 115.0  # 120s limit, keep 5s margin

    x0 = float(x0)
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)

    def within_budget():
        return time.time() < deadline

    # Track best |f| seen so far.
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = x
            best_f = fx
        return af

    # ------------------------------------------------------------------
    # Adaptive finite-difference derivative (central when possible).
    # Scales h with local scale and uses complex-step-free robust scheme.
    # ------------------------------------------------------------------
    def deriv(x, fx):
        scale = max(1.0, abs(x))
        # base step for central difference
        h = 1e-7 * scale
        # try central difference first (2 evals) for better accuracy
        try:
            f_plus = f(x + h)
            f_minus = f(x - h)
            d = (f_plus - f_minus) / (2.0 * h)
        except Exception:
            # fallback forward difference
            f_plus = f(x + h)
            d = (f_plus - fx) / h
        # If derivative is too small or non-finite, try a larger step.
        if not math.isfinite(d) or abs(d) < 1e-300:
            for h_try in (1e-5 * scale, 1e-4 * scale, 1e-3 * scale, 1e-2 * scale):
                try:
                    f_plus = f(x + h_try)
                    d = (f_plus - fx) / h_try
                    if math.isfinite(d) and abs(d) > 1e-300:
                        break
                except Exception:
                    continue
        return d

    # ------------------------------------------------------------------
    # Phase 1: Safeguarded global search (bisection + secant/Newton hybrid).
    # Build a bracket around x0, then shrink it while taking secant steps
    # that stay inside the bracket (Illinois/modified regula falsi).
    # ------------------------------------------------------------------
    # Initial bracket expansion.
    lo = hi = None
    f_lo = f_hi = None
    if math.isfinite(best_f):
        f0 = best_f
        # Expand outward geometrically to find a sign change.
        step = max(1.0, abs(x0)) * 0.5
        found = False
        for _ in range(40):
            if not within_budget():
                break
            if not found:
                # try x0 - step
                try:
                    xl = x0 - step
                    fxl = f(xl)
                except Exception:
                    fxl = float('nan')
                if math.isfinite(fxl) and f0 * fxl < 0:
                    lo, hi = xl, x0
                    f_lo, f_hi = fxl, f0
                    found = True
                    break
                # try x0 + step
                try:
                    xh = x0 + step
                    fxh = f(xh)
                except Exception:
                    fxh = float('nan')
                if math.isfinite(fxh) and f0 * fxh < 0:
                    lo, hi = x0, xh
                    f_lo, f_hi = f0, fxh
                    found = True
                    break
                step *= 2.0
            else:
                break

    if found:
        # We have a bracket [lo, hi] with f_lo * f_hi < 0.
        # Use a safeguarded Illinois + bisection hybrid.
        for _ in range(200):
            if not within_budget():
                break
            # Check endpoints.
            if abs(f_lo) <= 1e-10:
                best_x, best_f = lo, f_lo
                break
            if abs(f_hi) <= 1e-10:
                best_x, best_f = hi, f_hi
                break

            width = hi - lo
            if width <= 1e-12 * max(1.0, abs(lo), abs(hi)):
                # Bracket is tiny; pick midpoint.
                mid = 0.5 * (lo + hi)
                try:
                    fm = f(mid)
                except Exception:
                    fm = float('nan')
                if math.isfinite(fm):
                    update_best(mid, fm)
                break

            # Secant step (regula falsi).
            denom = f_hi - f_lo
            if abs(denom) > 1e-300:
                x_s = hi - f_hi * (hi - lo) / denom
            else:
                x_s = 0.5 * (lo + hi)

            # Safeguard: keep the secant point inside the bracket.
            if not (lo < x_s < hi):
                x_s = 0.5 * (lo + hi)

            try:
                f_s = f(x_s)
            except Exception:
                f_s = float('nan')

            if not math.isfinite(f_s):
                # Fall back to bisection.
                x_s = 0.5 * (lo + hi)
                try:
                    f_s = f(x_s)
                except Exception:
                    break
                if not math.isfinite(f_s):
                    break

            update_best(x_s, f_s)
            if abs(f_s) <= 1e-10:
                best_x, best_f = x_s, f_s
                break

            # Update bracket.
            if f_lo * f_s < 0:
                hi, f_hi = x_s, f_s
            else:
                lo, f_lo = x_s, f_s

            # Illinois damping: shrink the side we keep to avoid slow convergence.
            if f_lo * f_hi > 0:
                # Should not happen, but guard.
                break

    # ------------------------------------------------------------------
    # Phase 2: High-precision local polish (Newton with adaptive FD).
    # Start from the best point found; use tighter tolerance.
    # ------------------------------------------------------------------
    x = best_x
    fx = best_f
    for _ in range(80):
        if not within_budget():
            break
        af = abs(fx)
        if af <= 1e-12:
            break
        d = deriv(x, fx)
        if not math.isfinite(d) or abs(d) < 1e-300:
            break
        step = fx / d
        # Limit step size to avoid diverging in ill-conditioned regions.
        max_step = 1e-3 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
            break
        try:
            fx_new = f(x_new)
        except Exception:
            break
        if not math.isfinite(fx_new):
            break
        af_new = abs(fx_new)
        if af_new < af:
            x, fx = x_new, fx_new
            update_best(x, fx)
        else:
            # Step made things worse; halve and retry a couple times.
            improved = False
            for _halve in range(3):
                step *= 0.5
                x_try = x - step
                if not math.isfinite(x_try):
                    break
                try:
                    fx_try = f(x_try)
                except Exception:
                    break
                if not math.isfinite(fx_try):
                    break
                if abs(fx_try) < af:
                    x, fx = x_try, fx_try
                    update_best(x, fx)
                    improved = True
                    break
            if not improved:
                break

    # ------------------------------------------------------------------
    # Phase 3: Final ultra-tight polish (few iterations, very tight tol).
    # ------------------------------------------------------------------
    x = best_x
    fx = best_f
    for _ in range(40):
        if not within_budget():
            break
        af = abs(fx)
        if af <= 1e-14:
            break
        d = deriv(x, fx)
        if not math.isfinite(d) or abs(d) < 1e-300:
            break
        step = fx / d
        if not math.isfinite(step):
            break
        x_new = x - step
        if not math.isfinite(x_new):
            break
        try:
            fx_new = f(x_new)
        except Exception:
            break
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) < af:
            x, fx = x_new, fx_new
            update_best(x, fx)
        else:
            break

    # Return the best x found (guarantees minimal |f| within our budget).
    return best_x
# EVOLVE-BLOCK-END
