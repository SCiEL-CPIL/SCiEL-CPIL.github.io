"""Root finding: Ill-conditioned - Illinois method with adaptive bracketing."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Illinois method with adaptive step-doubling for bracketing,
    specifically designed for ill-conditioned functions.
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
        except (ValueError, OverflowError, ZeroDivisionError):
            return float('inf')
        if not math.isfinite(val):
            return float('inf')
        return val

    def is_good(fx):
        return abs(fx) <= TARGET_TOL

    def check_time():
        return (time.time() - start_time) < MAX_WALL and eval_count[0] < MAX_EVALS

    def illinois_step(a, fa, b, fb, shrink=0.5):
        """
        Illinois method step: modified secant that shrinks the larger sub-interval.
        Returns (new_a, new_fa, new_b, new_fb) or None if unable to proceed.
        """
        if not check_time():
            return None
        if not math.isfinite(fa) or not math.isfinite(fb):
            return None
        if fa * fb > 0:
            return None

        # Standard secant point
        denom = fb - fa
        if abs(denom) < 1e-15:
            # Fall back to bisection
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if not math.isfinite(fm):
                return None
            if is_good(fm):
                return (a, fa, mid, fm) if fa * fm <= 0 else (mid, fm, b, fb)
            if fa * fm <= 0:
                return (a, fa, mid, fm)
            else:
                return (mid, fm, b, fb)

        x_secant = b - fb * (b - a) / denom

        # Illinois modification: if x_secant is outside [a,b] or on the same side,
        # shrink the larger sub-interval
        if x_secant <= a or x_secant >= b:
            # Use weighted midpoint that shrinks the larger interval
            if abs(fa) < abs(fb):
                # Shrink [a, mid] side
                x_new = a + (b - a) * (1 - shrink)
            else:
                # Shrink [mid, b] side
                x_new = b - (b - a) * (1 - shrink)
        else:
            x_new = x_secant

        # Ensure x_new is strictly inside (a, b)
        eps = 1e-14 * max(1.0, abs(a), abs(b))
        if x_new <= a + eps:
            x_new = a + eps
        if x_new >= b - eps:
            x_new = b - eps

        fx_new = eval_f(x_new)
        if not math.isfinite(fx_new):
            # Fall back to bisection
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if not math.isfinite(fm):
                return None
            if is_good(fm):
                return (a, fa, mid, fm) if fa * fm <= 0 else (mid, fm, b, fb)
            if fa * fm <= 0:
                return (a, fa, mid, fm)
            else:
                return (mid, fm, b, fb)

        if is_good(fx_new):
            return (a, fa, x_new, fx_new) if fa * fx_new <= 0 else (x_new, fx_new, b, fb)

        if fa * fx_new <= 0:
            return (a, fa, x_new, fx_new)
        else:
            return (x_new, fx_new, b, fb)

    def find_bracket_adaptive(x, fx, max_steps=30):
        """
        Adaptive step-doubling to find a sign-change bracket around x.
        Tries expanding in both directions.
        """
        if not math.isfinite(fx):
            return None

        # Try expanding from x in both directions
        h = 1e-4 * max(1.0, abs(x))
        
        for step in range(max_steps):
            if not check_time():
                return None
            
            # Check left expansion
            x_left = x - h
            fl = eval_f(x_left)
            if math.isfinite(fl) and fl * fx < 0:
                return (x_left, fl, x, fx)
            
            # Check right expansion
            x_right = x + h
            fr = eval_f(x_right)
            if math.isfinite(fr) and fr * fx < 0:
                return (x, fx, x_right, fr)
            
            # Double the step size
            h *= 2.0
            if h > 1e6 * max(1.0, abs(x)):
                break

        return None

    def polish(x, fx, max_iters=10):
        """Final polish using secant/Newton-like steps."""
        for _ in range(max_iters):
            if not check_time():
                break
            if abs(fx) <= 1e-12:
                break
            
            h = 1e-8 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if not math.isfinite(fx_plus):
                break
            
            df = (fx_plus - fx) / h
            if abs(df) < 1e-15:
                break
            
            step = fx / df
            max_step = 1.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * math.copysign(1.0, step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                break
        
        return x, fx

    # Main algorithm
    x = float(x0)
    fx = eval_f(x)
    best_x = x
    best_fx = fx

    if is_good(fx):
        return x

    # Phase 1: Try to find a bracket using adaptive step-doubling
    bracket = find_bracket_adaptive(x, fx)
    
    if bracket is not None:
        a, fa, b, fb = bracket
        # Phase 2: Apply Illinois method on the bracket
        for _ in range(300):
            if not check_time():
                break
            
            # Check if we've found a good solution
            if is_good(fa):
                best_x, best_fx = a, fa
                break
            if is_good(fb):
                best_x, best_fx = b, fb
                break
            
            # Check interval width
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                # Interval is tiny; return midpoint
                mid = (a + b) / 2.0
                fm = eval_f(mid)
                if math.isfinite(fm) and abs(fm) < abs(best_fx):
                    best_x, best_fx = mid, fm
                break
            
            # Take Illinois step
            result = illinois_step(a, fa, b, fb)
            if result is None:
                break
            
            a, fa, b, fb = result
            
            # Track best
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

    # Phase 3: If no bracket was found or Illinois didn't converge, try local search
    if not is_good(best_fx) and check_time():
        # Try secant method from x0 with a perturbed point
        h = 1e-4 * max(1.0, abs(x))
        x2 = x + h
        fx2 = eval_f(x2)
        
        if math.isfinite(fx2):
            x1, fx1 = x, fx
            for _ in range(100):
                if not check_time():
                    break
                
                if is_good(fx1):
                    best_x, best_fx = x1, fx1
                    break
                if is_good(fx2):
                    best_x, best_fx = x2, fx2
                    break
                
                dx = x2 - x1
                if abs(dx) < 1e-15:
                    break
                
                df = (fx2 - fx1) / dx
                if abs(df) < 1e-15:
                    break
                
                step = fx2 / df
                max_step = 10.0 * max(1.0, abs(x2))
                if abs(step) > max_step:
                    step = max_step * math.copysign(1.0, step)
                
                x_new = x2 - step
                fx_new = eval_f(x_new)
                
                if not math.isfinite(fx_new):
                    break
                
                if abs(fx_new) < abs(best_fx):
                    best_x, best_fx = x_new, fx_new
                
                # Shift the pair
                x1, fx1 = x2, fx2
                x2, fx2 = x_new, fx_new
                
                if is_good(fx2):
                    best_x, best_fx = x2, fx2
                    break

    # Phase 4: Final polish
    if check_time() and not is_good(best_fx):
        x_pol, fx_pol = polish(best_x, best_fx)
        if math.isfinite(fx_pol) and abs(fx_pol) < abs(best_fx):
            best_x, best_fx = x_pol, fx_pol

    return best_x

# EVOLVE-BLOCK-END
