"""Root finding via global adaptive grid search + secant polishing.

Instead of starting a local Newton iteration at x0 (which can diverge on
ill-conditioned problems), we first perform a coarse-to-fine logarithmic grid
sweep around x0 to find the point with the smallest |f| in a wide neighborhood.
This global scan is robust to local ill-conditioning because it does not rely
on gradient information. Once the best grid point is identified, we polish
with a pure secant method (no derivative evaluation needed), which is cheap
(1 eval/step) and superlinear near simple roots. A bisection fallback is used
if a sign change is detected during polishing.
"""
import math
import time

_MAX_EVALS = 500
_DEADLINE = 110.0  # safety margin before 120s limit


def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.

    Returns x (float).
    """
    x0 = float(x0)
    count = [0]
    t_start = time.time()
    best_x = x0
    best_fabs = float('inf')

    def eval_f(x):
        """Evaluate f(x) with budget tracking."""
        nonlocal best_x, best_fabs
        if count[0] >= _MAX_EVALS:
            return None
        if time.time() - t_start > _DEADLINE:
            return None
        try:
            val = float(f(x))
        except Exception:
            return None
        count[0] += 1
        if math.isfinite(val) and abs(val) < best_fabs:
            best_fabs = abs(val)
            best_x = x
        return val

    # Evaluate at x0
    fx0 = eval_f(x0)
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0

    if fx0 is None:
        return x0

    # --- Phase 1: Adaptive logarithmic grid search ---
    # We scan a logarithmic grid of scales around x0 to find where |f| is small.
    # This handles cases where x0 is far from the root or in a region of
    # ill-conditioning (flat regions, multiple roots, etc.)
    
    def grid_search(center, scale, n_points=20):
        """Scan a grid around center with given scale. Returns best (x, fx)."""
        local_best_x = None
        local_best_fabs = float('inf')
        
        # Generate logarithmically spaced points from -scale to +scale
        # Use symmetric points: center ± scale * (1, 2, 4, ..., 2^(n-1))
        # Plus center itself
        candidates = [center]
        for i in range(n_points):
            d = scale * (2 ** i)
            candidates.append(center + d)
            candidates.append(center - d)
        
        # Also add some intermediate points for better resolution
        for i in range(1, n_points):
            d = scale * (2 ** i) * 0.5
            candidates.append(center + d)
            candidates.append(center - d)
        
        for x in candidates:
            if count[0] >= _MAX_EVALS or time.time() - t_start > _DEADLINE:
                break
            fx = eval_f(x)
            if fx is not None and abs(fx) < local_best_fabs:
                local_best_fabs = abs(fx)
                local_best_x = x
        
        return local_best_x, local_best_fabs

    # Start with a moderate scale and expand if needed
    base_scale = max(1.0, abs(x0)) * 0.1
    current_best_x = x0
    current_best_fabs = abs(fx0)
    
    for scale_factor in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        if count[0] >= _MAX_EVALS - 50 or time.time() - t_start > _DEADLINE - 5.0:
            break
        
        scale = base_scale * scale_factor
        # Limit number of points based on remaining budget
        remaining = _MAX_EVALS - count[0]
        n_pts = min(20, max(4, remaining // 3))
        
        bx, bf = grid_search(current_best_x, scale, n_pts)
        if bx is not None and bf < current_best_fabs:
            current_best_x = bx
            current_best_fabs = bf
            if current_best_fabs <= 1e-8:
                return current_best_x
        
        # If we found a much better point, re-center the grid search
        if bf < current_best_fabs * 0.1:
            # Continue expanding from this new center
            pass
        elif scale_factor >= 100.0:
            # If we've expanded a lot without improvement, stop
            break

    # --- Phase 2: Secant polishing from the best point found ---
    # Use the secant method to refine. We need two points; use the best point
    # and a nearby point.
    
    x1 = current_best_x
    f1 = eval_f(x1)
    if f1 is None:
        return best_x
    if abs(f1) <= 1e-8:
        return x1

    # Get a second point for the secant method
    # Try a small perturbation
    h = max(1e-6, abs(x1) * 1e-6)
    x2 = x1 + h
    f2 = eval_f(x2)
    if f2 is None:
        # Try the other direction
        x2 = x1 - h
        f2 = eval_f(x2)
    if f2 is None:
        return best_x

    # Secant iteration
    for _ in range(100):
        if count[0] >= _MAX_EVALS - 5 or time.time() - t_start > _DEADLINE:
            break
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2

        # Check for sign change -> use bisection
        if f1 * f2 < 0:
            # Bisection
            a, b = (x1, x2) if f1 * f2 < 0 else (x2, x1)
            fa, fb = f1, f2
            for _ in range(60):
                if count[0] >= _MAX_EVALS - 2 or time.time() - t_start > _DEADLINE:
                    break
                mid = (a + b) / 2.0
                fmid = eval_f(mid)
                if fmid is None:
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                    break
            # Return the best of a, b
            if abs(fa) <= abs(fb):
                return a
            else:
                return b

        # Secant step
        if abs(f2 - f1) < 1e-15:
            # Degenerate: f values too close, perturb
            x2 = x1 + h * 10
            f2 = eval_f(x2)
            if f2 is None:
                break
            continue

        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        
        # Safeguard: don't go too far
        max_step = max(1.0, abs(x1)) * 100
        if abs(x_new - x1) > max_step:
            x_new = x1 + math.copysign(max_step, x_new - x1)

        f_new = eval_f(x_new)
        if f_new is None:
            break

        # Update the two points: keep the one with smaller |f|
        if abs(f_new) < abs(f1):
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        else:
            x2, f2 = x_new, f_new

        # Convergence check
        if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
            break

    # Return the best point found
    return best_x
