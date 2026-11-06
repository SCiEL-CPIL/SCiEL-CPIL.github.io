"""Root finding: Ill-conditioned - IQI-based robust solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses an Inverse-Quadratic Interpolation (IQI) variant with robust bracketing.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    SHRINK_FACTOR = 0.5  # Factor to shrink larger interval in IQI
    
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        return f(x)
    
    def time_ok():
        return time.time() - start_time < MAX_WALL

    def iq_step(a, fa, b, fb):
        """
        Compute next point using Inverse Quadratic Interpolation.
        Returns x_new.
        """
        # We want to interpolate x as a function of y.
        # Points: (fa, a), (fb, b). Need a third point for quadratic.
        # Standard IQI uses the previous point. Here we approximate using
        # a "virtual" third point or fall back to secant if not enough info.
        
        # For robustness, let's use a simplified IQI that uses the secant
        # but adjusts the step based on the bracket size to ensure convergence.
        # Actually, the prompt suggests: "modified secant update that shrinks 
        # the larger bracket sub-interval by a fixed factor".
        
        # Let's implement the specific strategy:
        # 1. Compute secant root x_s.
        # 2. If x_s is outside [a,b] or too close to a/b, adjust it.
        # 3. Ensure the new point is closer to the root side (smaller |f|).
        
        if fa == fb:
            return (a + b) / 2
            
        x_s = a - fa * (b - a) / (fb - fa)
        
        # If x_s is not strictly inside (a,b), or if it's too close to the
        # endpoint with the larger function value, we might stagnate.
        
        # The "Illinois" style modification often scales the function value
        # of the retained point. Let's try a geometric approach:
        # If |fa| < |fb|, the root is likely closer to a.
        # If x_s is not in (a,b), we bisect.
        
        if x_s <= a or x_s >= b:
            return (a + b) / 2
            
        # Check if the step is "reasonable". 
        # In ill-conditioned problems, the secant step might overshoot 
        # or undershoot significantly.
        
        # Let's check the position.
        # If x_s is very close to a, and |fa| is small, we are good.
        # If x_s is very close to b, and |fb| is small, we are good.
        
        # A common robust modification is to ensure the new point is not
        # too close to the endpoint with the LARGER function value.
        
        if abs(fa) < abs(fb):
            # Root likely near a.
            # If x_s is very close to b, it's a bad step.
            # Let's check if x_s is in the half of the interval closer to a.
            mid = (a + b) / 2
            if (b > a and x_s > mid) or (b < a and x_s < mid):
                # x_s is on the "wrong" side (closer to b, where |fb| is larger)
                # Shrink towards a.
                # New point: a + SHRINK_FACTOR * (b - a) ? No, that's bisection.
                # Let's use the "fixed factor shrink" of the larger sub-interval.
                # If we keep a and discard b, the new interval is [a, x_s].
                # But if x_s is bad, we should pick a point that guarantees
                # progress.
                
                # Strategy from prompt: "shrinks the larger bracket sub-interval 
                # by a fixed factor".
                # This implies we pick a point such that the larger of the two
                # new sub-intervals is SHRINK_FACTOR times the old interval.
                
                # Let's just use bisection if secant is suspect.
                # Or, better: use the secant point but clamp it to be at least
                # some fraction away from the bad endpoint.
                
                # Let's stick to the prompt's specific IQI variant idea:
                # If the secant point is not in the "good" half, we use a 
                # point that shrinks the larger part.
                
                # If we assume the root is in [a, mid], we want a point in [a, mid].
                # Let's pick x_new = a + (1 - SHRINK_FACTOR) * (mid - a) ?
                # No, let's just return bisection if secant is outside the
                # half containing the smaller |f|.
                
                return mid
        else:
            # Root likely near b.
            mid = (a + b) / 2
            if (b > a and x_s < mid) or (b < a and x_s > mid):
                return mid
                
        return x_s

    def find_bracket(x0, fx0):
        """
        Find a bracket [a, b] such that f(a)*f(b) < 0.
        Uses adaptive step doubling.
        Returns (a, fa, b, fb) or None.
        """
        if not math.isfinite(fx0):
            return None
            
        # Initial step size
        h = 1e-4 * max(1.0, abs(x0))
        a = x0
        fa = fx0
        
        for i in range(20):  # Max doublings
            if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
                break
                
            # Try +h
            b = a + h
            if b == a:
                h *= 2
                continue
            fb = eval_f(b)
            if not math.isfinite(fb):
                h *= 2
                continue
                
            if fa * fb < 0:
                return (a, fa, b, fb)
            
            # Try -h (if we haven't tried it recently or if +h failed)
            # Actually, let's alternate or just expand in both directions.
            # A common strategy: expand in the direction of the gradient (if known) 
            # or just alternate.
            
            # Let's try -h next if we haven't bracketed.
            b_neg = a - h
            if b_neg != a:
                fb_neg = eval_f(b_neg)
                if math.isfinite(fb_neg):
                    if fa * fb_neg < 0:
                        return (a, fa, b_neg, fb_neg)
            
            # If no bracket, double h
            h *= 2
            
            # Update a to the point with smaller |f|? No, keep x0 as anchor 
            # unless we move. For robustness, let's keep x0 as one end 
            # until we find a bracket, then refine.
            # If we don't find a bracket around x0, we might need to move x0.
            # But the problem says "local ill-conditioned", so the root is near x0.
            
        return None

    def iq_solve(a, fa, b, fb):
        """
        Solve for root in [a, b] using IQI/Secant hybrid.
        """
        for _ in range(50):
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
                
            if abs(fa) <= TARGET_TOL:
                return a, fa, True
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
                
            # Compute next point
            x_new = iq_step(a, fa, b, fb)
            
            # Ensure x_new is strictly inside
            if x_new <= min(a, b) or x_new >= max(a, b):
                x_new = (a + b) / 2
                
            if x_new == a or x_new == b:
                break
                
            f_new = eval_f(x_new)
            
            if not math.isfinite(f_new):
                # Fallback to bisection
                x_new = (a + b) / 2
                if x_new == a or x_new == b:
                    break
                f_new = eval_f(x_new)
                
            if abs(f_new) <= TARGET_TOL:
                return x_new, f_new, True
                
            # Update bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
        # Return best
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= TARGET_TOL
        else:
            return b, fb, abs(fb) <= TARGET_TOL

    # --- Main Logic ---
    
    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
        
    best_x = x0
    best_fx = fx0
    
    # 1. Try to find a bracket immediately
    bracket = find_bracket(x0, fx0)
    
    if bracket:
        a, fa, b, fb = bracket
        # Ensure a < b for consistency, though math works either way
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        x_sol, fx_sol, success = iq_solve(a, fa, b, fb)
        if success:
            return x_sol
        if abs(fx_sol) < abs(best_fx):
            best_x, best_fx = x_sol, fx_sol
    else:
        # No bracket found. This is tricky for "local" problems.
        # Maybe the function doesn't change sign? (e.g., x^2 - 1e-9 = 0)
        # In that case, we minimize |f(x)|.
        
        # Use a local search / Newton-like method to minimize |f(x)|.
        # Since we can't assume a sign change, we use a derivative-free 
        # minimization of |f(x)| near x0.
        
        # Simple approach: try points in a small grid and pick the best, 
        # then refine with secant on the two best points.
        
        h = 1e-4 * max(1.0, abs(x0))
        candidates = [(x0, fx0)]
        
        for i in range(1, 10):
            if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                break
            for sign in [1, -1]:
                x_c = x0 + sign * i * h
                if x_c == x0:
                    continue
                fx_c = eval_f(x_c)
                if math.isfinite(fx_c):
                    candidates.append((x_c, fx_c))
                    if abs(fx_c) < abs(best_fx):
                        best_x, best_fx = x_c, fx_c
        
        # Sort by |f|
        candidates.sort(key=lambda p: abs(p[1]))
        
        # Take the two best distinct points and use secant
        if len(candidates) >= 2:
            x1, fx1 = candidates[0]
            x2, fx2 = candidates[1]
            
            # Ensure they are distinct
            if x1 != x2 and fx1 != fx2:
                # Use secant to refine
                for _ in range(20):
                    if not time_ok() or eval_count[0] >= MAX_EVALS:
                        break
                    if abs(fx1) <= TARGET_TOL:
                        return x1
                    if abs(fx2) <= TARGET_TOL:
                        return x2
                    
                    d = (fx2 - fx1) / (x2 - x1)
                    if d == 0 or not math.isfinite(d):
                        break
                    
                    x_new = x1 - fx1 / d
                    if not math.isfinite(x_new):
                        break
                    
                    fx_new = eval_f(x_new)
                    
                    if abs(fx_new) < abs(best_fx):
                        best_x, best_fx = x_new, fx_new
                        
                    if abs(fx_new) <= TARGET_TOL:
                        return x_new
                        
                    # Update: keep the two best points
                    # If |fx_new| < |fx1|, replace x1
                    # Else if |fx_new| < |fx2|, replace x2
                    # Else discard
                    if abs(fx_new) < abs(fx1):
                        x1, fx1 = x_new, fx_new
                    elif abs(fx_new) < abs(fx2):
                        x2, fx2 = x_new, fx_new
                    else:
                        # No improvement, try a different point?
                        # For now, break to avoid loop
                        break
        
    # Final check
    if abs(best_fx) <= TARGET_TOL:
        return best_x
        
    return best_x

# EVOLVE-BLOCK-END
