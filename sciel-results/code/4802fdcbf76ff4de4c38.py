"""Root finder for 3 fixed oscillatory equations using bracketing + Illinois method."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit
    MAX_CALLS = 490     # Stay well under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = float(x)
        return float(val)

    def _squared_residual(x):
        """Wrapper for minimize_scalar: returns f(x)^2."""
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() - start_time > TIME_LIMIT:
            return float('inf')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('inf')
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return float('inf')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = float(x)
        return float(val) * float(val)

    def brent_like(fa, a, fb, b):
        """Brent-like method with safeguarded secant + bisection."""
        nonlocal calls
        for _ in range(300):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Try secant step
            if fa != fb and (fb - fa) != 0.0:
                secant_x = (a * fb - b * fa) / (fb - fa)
            else:
                secant_x = (a + b) / 2.0
            
            # Safeguard: if secant is outside or too close to endpoint, use bisection
            if not math.isfinite(secant_x) or not (a < secant_x < b):
                secant_x = (a + b) / 2.0
            
            # If secant is very close to an endpoint, use bisection
            scale = max(1.0, abs(a), abs(b))
            if abs(secant_x - a) < 1e-12 * scale or abs(secant_x - b) < 1e-12 * scale:
                secant_x = (a + b) / 2.0
            
            fs = _f(secant_x)
            if math.isnan(fs):
                break
            if abs(fs) <= 1e-8:
                return secant_x
            
            if fa * fs < 0:
                b = secant_x
                fb = fs
            else:
                a = secant_x
                fa = fs
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    def illinois(fa, a, fb, b):
        """Illinois method: bisection + linear interpolation with damping."""
        nonlocal calls
        for _ in range(300):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            # Illinois: linear interpolation, but if it's outside [a,b], use midpoint
            if fa == fb:
                mid = (a + b) / 2.0
            else:
                mid = (a * fb - b * fa) / (fb - fa)
            if not math.isfinite(mid) or not (a < mid < b):
                mid = (a + b) / 2.0
            fm = _f(mid)
            if math.isnan(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
                fa = fa / 2.0
            else:
                a = mid
                fa = fm
                fb = fb / 2.0
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # Phase 1: Evaluate at x0
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Phase 2: Try expanding in both directions to find a sign change
    found_bracket = False
    best_bracket = None  # (fa, a, fb, b) with smallest |fa|+|fb|
    min_expanded_x = x
    max_expanded_x = x

    # Try positive direction with adaptive step
    step = 0.05 * max(1.0, abs(x))
    for i in range(1, 80):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_pos = x + i * step
        f_pos = _f(x_pos)
        if math.isnan(f_pos):
            break
        if abs(f_pos) <= 1e-8:
            return x_pos
        if fx * f_pos < 0:
            # Found bracket [x, x_pos]
            found_bracket = True
            result = brent_like(fx, x, f_pos, x_pos)
            return result
        step *= 1.3  # Expand adaptively
        max_expanded_x = x_pos

    # Try negative direction with adaptive step
    step = 0.05 * max(1.0, abs(x))
    for i in range(1, 80):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_neg = x - i * step
        f_neg = _f(x_neg)
        if math.isnan(f_neg):
            break
        if abs(f_neg) <= 1e-8:
            return x_neg
        if fx * f_neg < 0:
            # Found bracket [x_neg, x]
            found_bracket = True
            result = brent_like(f_neg, x_neg, fx, x)
            return result
        step *= 1.3
        min_expanded_x = x_neg

    # Phase 3: If no bracket found, do a finer local scan
    # Try a dense local grid around x0 with multiple scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        local_step = scale * max(1.0, abs(x))
        n_points = min(21, max(5, int(490 - calls - 10)))
        if n_points < 3:
            break
        prev_x = x - (n_points - 1) * local_step / 2
        prev_fx = _f(prev_x)
        if math.isnan(prev_fx):
            continue
        if abs(prev_fx) <= 1e-8:
            return prev_x

        for i in range(1, n_points):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = x - (n_points - 1) * local_step / 2 + i * local_step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return cur_x
            if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
                result = brent_like(prev_fx, prev_x, cur_fx, cur_x)
                return result
            prev_x = cur_x
            prev_fx = cur_fx

    # Phase 4: Try Illinois method on best available bracket if we have one
    # Re-scan with smaller steps to find any sign change
    for scale in [0.005, 0.02, 0.08, 0.3, 1.5, 3.0]:
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        local_step = scale * max(1.0, abs(x))
        n_points = min(15, max(4, int(490 - calls - 5)))
        if n_points < 3:
            break
        prev_x = x - (n_points - 1) * local_step / 2
        prev_fx = _f(prev_x)
        if math.isnan(prev_fx):
            continue
        if abs(prev_fx) <= 1e-8:
            return prev_x

        for i in range(1, n_points):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = x - (n_points - 1) * local_step / 2 + i * local_step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return cur_x
            if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
                result = illinois(prev_fx, prev_x, cur_fx, cur_x)
                return result
            prev_x = cur_x
            prev_fx = cur_fx

    # Phase 5: Use SciPy minimize_scalar as a fallback optimization approach
    if calls < MAX_CALLS - 20 and time.time() - start_time < TIME_LIMIT - 5:
        try:
            from scipy.optimize import minimize_scalar
            
            # Determine search bounds from exploration
            span = max(abs(max_expanded_x - min_expanded_x), 1.0)
            center = (max_expanded_x + min_expanded_x) / 2.0
            lower = min(min_expanded_x, x - span)
            upper = max(max_expanded_x, x + span)
            
            # Ensure bounds are finite and reasonable
            if not (math.isfinite(lower) and math.isfinite(upper)):
                lower = x - 10.0
                upper = x + 10.0
            
            # Run minimize_scalar with bounded method
            result = minimize_scalar(
                _squared_residual,
                bounds=(lower, upper),
                method='bounded',
                options={'maxiter': 100, 'xatol': 1e-10}
            )
            
            if result.success and math.isfinite(result.x):
                # Verify the result
                if abs(best_abs_f) <= 1e-8:
                    return best_x
                # Do a quick polish around the minimizer
                polish_x = float(result.x)
                polish_f = _f(polish_x)
                if not math.isnan(polish_f) and abs(polish_f) <= 1e-8:
                    return polish_x
                # Try a few nearby points
                for dx in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4]:
                    if calls >= MAX_CALLS:
                        break
                    px = polish_x + dx
                    pf = _f(px)
                    if not math.isnan(pf) and abs(pf) < best_abs_f:
                        best_x = px
                        best_abs_f = abs(pf)
                    if abs(pf) <= 1e-8:
                        return px
        except Exception:
            pass

    # Final polish: if best_abs_f is close, try to tighten
    if best_abs_f < 1e-6 and calls < MAX_CALLS - 10:
        # Do a local Newton-like step using finite differences
        for _ in range(10):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            h = 1e-8 * max(1.0, abs(best_x))
            f_left = _f(best_x - h)
            f_right = _f(best_x + h)
            if math.isnan(f_left) or math.isnan(f_right):
                break
            deriv = (f_right - f_left) / (2.0 * h)
            if abs(deriv) < 1e-15:
                break
            new_x = best_x - best_abs_f * math.copysign(1.0, f(best_x) if not math.isnan(_f(best_x)) else 1.0) / deriv
            if not math.isfinite(new_x):
                break
            nf = _f(new_x)
            if math.isnan(nf):
                break
            if abs(nf) < best_abs_f:
                best_x = new_x
                best_abs_f = abs(nf)
            if abs(nf) <= 1e-8:
                return new_x

    # Return best x found
    return best_x

# EVOLVE-BLOCK-END
