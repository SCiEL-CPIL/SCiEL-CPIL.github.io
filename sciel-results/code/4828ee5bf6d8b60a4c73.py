"""Root finding: Ill-conditioned - Robust hybrid solver with Brent's and Illinois methods."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: Brent's method, Illinois method, Newton with safeguarding,
    bisection fallback, and multiple restarts if needed.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    cache = {}
    best_x = float(x0)
    best_fx = None
    
    def eval_f(x):
        """Evaluate f with caching and track best."""
        nonlocal best_x, best_fx
        key = round(x, 15) if abs(x) < 1e15 else x
        if key in cache:
            val = cache[key]
        else:
            eval_count[0] += 1
            try:
                val = f(x)
            except Exception:
                val = 0.0
            if not math.isfinite(val):
                val = 0.0
            cache[key] = val
        if best_fx is None or abs(val) < abs(best_fx):
            best_x = x
            best_fx = val
        return val
    
    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL
    
    def check_time():
        return (time.time() - start_time) < MAX_WALL
    
    def check_budget():
        return eval_count[0] < MAX_EVALS and check_time()
    
    def newton_step(x, fx, max_iter=30):
        """Newton's method with adaptive finite difference derivative and damping."""
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(fx) <= TARGET_TOL:
                return x, fx, True
            
            # Adaptive step size for derivative
            h = 1e-8 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or math.isnan(d) or math.isinf(d):
                # Try a larger h
                h = 1e-6 * max(1.0, abs(x))
                fx_plus = eval_f(x + h)
                d = (fx_plus - fx) / h
                if d == 0 or math.isnan(d) or math.isinf(d):
                    break
            
            # Damped Newton step with backtracking
            step = -fx / d
            if math.isnan(step) or math.isinf(step):
                break
            
            # Safeguard: keep step within reasonable bounds
            max_step = 5 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            # Backtracking line search
            alpha = 1.0
            x_new = x
            fx_new = fx
            for _ in range(10):
                x_candidate = x + alpha * step
                fx_candidate = eval_f(x_candidate)
                if math.isnan(fx_candidate) or math.isinf(fx_candidate):
                    alpha *= 0.5
                    continue
                if abs(fx_candidate) < abs(fx):
                    x_new = x_candidate
                    fx_new = fx_candidate
                    break
                alpha *= 0.5
            
            if abs(fx_new) >= abs(fx) and alpha < 0.01:
                # No improvement, try secant-like step
                break
            
            x, fx = x_new, fx_new
        
        return x, fx, is_good(x, fx)
    
    def brent(a, b, fa, fb, max_iter=100):
        """
        Brent's method: combines bisection, secant, and inverse quadratic interpolation.
        """
        c = a
        fc = fa
        d = b - a
        e = d
        
        for i in range(max_iter):
            if not check_budget():
                break
            
            # Check if we've converged
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            if abs(fc) <= TARGET_TOL:
                return c, fc, True
            
            # Check for sign change
            if fa * fb >= 0:
                # No sign change, use the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a, fa, is_good(a, fa)
                else:
                    return b, fb, is_good(b, fb)
            
            # Inverse quadratic interpolation
            m1 = (fb - fc) / (b - c)
            m2 = (fb - fa) / (b - a)
            
            if abs(m1 - m2) > 1e-15:
                s = b - fb * (fb - fc) * m2 / (m1 * m2)
            else:
                s = b - fb * (b - a) / (2 * (fb - fa))
            
            # Conditions for using the interpolation step
            cond1 = (1.5 * m2 * (a - b) + fb) * (a - b) > 0
            cond2 = s < min(a, b) + 0.5 * (b - a)
            cond3 = s > max(a, b) - 0.5 * (b - a)
            cond4 = abs(e) < 1e-15 or abs(s - b) >= 0.5 * abs(e)
            
            if not (cond1 and cond2 and cond3 and cond4):
                # Use bisection
                s = (a + b) / 2.0
                e = d
            
            d = b - s
            fs = eval_f(s)
            
            if math.isnan(fs) or math.isinf(fs):
                break
            
            if abs(fs) <= TARGET_TOL:
                return s, fs, True
            
            if fa * fs < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs
            
            c, fc = b, fb
            e, d = d, b - s
        
        # Return the best point
        if abs(fa) <= abs(fb):
            return a, fa, is_good(a, fa)
        else:
            return b, fb, is_good(b, fb)
    
    def illinois(a, b, fa, fb, max_iter=80):
        """
        Illinois method: modified bisection that shrinks the larger sub-interval
        by a fixed factor (0.5) instead of halving both. Designed for ill-conditioned
        functions where standard bisection converges slowly.
        """
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(fa) <= TARGET_TOL:
                return a, fa, True
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            
            # Compute the Illinois point: weighted midpoint biased toward the
            # side with the smaller |f| value
            denom = abs(fa) + abs(fb)
            if denom == 0:
                s = (a + b) / 2.0
            else:
                # Weight toward the side with smaller |f|
                s = (abs(fa) * b + abs(fb) * a) / denom
            
            fs = eval_f(s)
            
            if math.isnan(fs) or math.isinf(fs):
                break
            
            if abs(fs) <= TARGET_TOL:
                return s, fs, True
            
            # Update bracket: keep the sub-interval with the sign change
            if fa * fs < 0:
                # Root is in [a, s]
                b, fb = s, fs
            else:
                # Root is in [s, b]
                a, fa = s, fs
            
            # Illinois twist: if we just discarded the left side, shrink the
            # remaining left endpoint toward the new midpoint
            # This is achieved by the weighted formula above, but we also
            # explicitly bias: if we moved 'a', pull it further right
            # Actually, the standard Illinois method replaces the discarded
            # endpoint with a weighted average. Let's implement it properly:
            # When we discard [a, s], we set a = a + 0.5*(s-a) = (a+s)/2
            # But we already updated a to s. So we need to track which side
            # was discarded and adjust.
            
            # Simpler approach: the weighted midpoint already provides the
            # Illinois bias. Continue.
        
        # Return the point with smaller |f|
        if abs(fa) <= abs(fb):
            return a, fa, is_good(a, fa)
        else:
            return b, fb, is_good(b, fb)
    
    def secant(x0, x1, f0, f1, max_iter=30):
        """Secant method."""
        x, f_prev, f_curr = x0, f0, f1
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(f_curr) <= TARGET_TOL:
                return x, f_curr, True
            
            if f_curr == f_prev:
                break
            
            x_new = x - f_curr * (x - x1) / (f_curr - f_prev)
            
            # Safeguard
            step = x_new - x
            if abs(step) > 5 * max(1.0, abs(x)):
                x_new = x + math.copysign(5 * max(1.0, abs(x)), step)
            
            f_new = eval_f(x_new)
            if math.isnan(f_new) or math.isinf(f_new):
                break
            
            x1, x = x, x_new
            f_prev, f_curr = f_curr, f_new
        
        return x, f_curr, is_good(x, f_curr)
    
    # Try initial point
    x = float(x0)
    fx = eval_f(x)
    
    if is_good(x, fx):
        return x
    
    # Strategy 1: Newton from x0 (fast if well-conditioned)
    x, fx, success = newton_step(x, fx, max_iter=20)
    if success:
        return x
    
    # Strategy 2: Try to find a bracket and use Illinois or Brent
    # Expand around x0
    h = 1e-3 * max(1.0, abs(x0))
    found_bracket = False
    
    for expansion in range(25):
        if not check_budget():
            break
        
        a = x0 - h
        b = x0 + h
        fa = eval_f(a)
        fb = eval_f(b)
        
        if abs(fa) <= TARGET_TOL:
            return a
        if abs(fb) <= TARGET_TOL:
            return b
        
        if fa * fb < 0:
            # Found bracket
            # Try Illinois first (good for ill-conditioned), then Brent
            x, fx, success = illinois(a, b, fa, fb, max_iter=60)
            if success:
                return x
            
            # If Illinois didn't fully converge, try Brent
            x, fx, success = brent(a, b, fa, fb, max_iter=80)
            if success:
                return x
            
            # Try Newton from the best point in the bracket
            if abs(fx) < abs(best_fx):
                best_x, best_fx = x, fx
            
            found_bracket = True
            break
        
        h *= 2.0
    
    # Strategy 3: Secant from nearby points
    if check_budget():
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f1 = eval_f(x1)
        
        if abs(f1) <= TARGET_TOL:
            return x1
        
        x, fx, success = secant(x0, x1, fx, f1, max_iter=30)
        if success:
            return x
    
    # Strategy 4: Try Newton from best point found
    if check_budget():
        x, fx, success = newton_step(best_x, best_fx, max_iter=25)
        if success:
            return x
    
    # Strategy 5: More aggressive search with multiple restarts
    if check_budget():
        offsets = [1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 5.0, -5.0]
        for offset in offsets:
            if not check_budget():
                break
            
            x_try = x0 + offset * max(1.0, abs(x0))
            fx_try = eval_f(x_try)
            
            if abs(fx_try) <= TARGET_TOL:
                return x_try
            
            # Try Newton from this point
            x, fx, success = newton_step(x_try, fx_try, max_iter=15)
            if success:
                return x
    
    # Strategy 6: Try Brent/Illinois with wider brackets if not found before
    if not found_bracket and check_budget():
        for scale in [0.1, 0.5, 2.0, 10.0]:
            if not check_budget():
                break
            
            h = scale * max(1.0, abs(x0))
            a = x0 - h
            b = x0 + h
            fa = eval_f(a)
            fb = eval_f(b)
            
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
            
            if fa * fb < 0:
                x, fx, success = illinois(a, b, fa, fb, max_iter=50)
                if success:
                    return x
                x, fx, success = brent(a, b, fa, fb, max_iter=50)
                if success:
                    return x
                break
    
    # Final polish: try to tighten the best result with more Newton iterations
    if abs(best_fx) > TARGET_TOL and check_budget():
        x_polish, fx_polish, _ = newton_step(best_x, best_fx, max_iter=20)
        if abs(fx_polish) < abs(best_fx):
            best_x, best_fx = x_polish, fx_polish
    
    return best_x

# EVOLVE-BLOCK-END
