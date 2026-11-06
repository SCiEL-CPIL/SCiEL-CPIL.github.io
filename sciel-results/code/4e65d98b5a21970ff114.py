"""Root finding for ill-conditioned problems using structured interval sweep and safeguarded Newton."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a structured geometric sweep to bracket the root,
    followed by a safeguarded Newton-Secant method.
    
    Strategy:
    1. Geometric Sweep: Systematically evaluate f at points expanding geometrically 
       from x0 in both directions. This is deterministic and ensures we don't miss 
       a bracket due to random sampling.
    2. Bracketing: Identify an interval [a, b] where sign(f(a)) != sign(f(b)) or 
       where |f| is minimized.
    3. Refinement: Use a safeguarded Newton-Secant method (Illinois/Brent-like logic)
       to converge to |f(x)| <= 1e-8.
    4. Fallback: If no sign change is found, use the point with minimum |f| and 
       attempt local polishing.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    # State
    best_x = float(x0)
    best_fx = None
    
    try:
        best_fx = call_f(best_x)
    except Exception:
        best_fx = float('inf')
        
    if abs(best_fx) <= 1e-8:
        return best_x

    # Phase 1: Geometric Interval Sweep
    # We want to find a bracket [a, b] with a sign change, or a very good minimum.
    # We expand the search radius geometrically.
    
    # Initial scale
    scale = max(1e-6, abs(best_x) * 1e-6)
    # Maximum scale to avoid overflow or excessive search
    max_scale = max(1.0, abs(best_x)) * 1e6
    
    # Points evaluated in the sweep
    # We keep track of (x, fx) to find sign changes
    points = [(best_x, best_fx)]
    
    # Expand left and right
    # We alternate expansion to keep the search balanced
    left_x = best_x
    right_x = best_x
    
    # We will do a few rounds of expansion
    # Round 0 is already done (x0)
    
    # To find a bracket, we need points with opposite signs.
    # If no sign change is found, we look for the minimum |f|.
    
    # Let's perform a structured sweep
    # We'll evaluate points at: x0, x0 +/- scale, x0 +/- 2*scale, ... but geometric is better for ill-conditioned?
    # Geometric: x0, x0 +/- scale, x0 +/- scale*r, ...
    # Let's use a factor of 2 for simplicity and robustness.
    
    r = 2.0
    current_scale = scale
    
    # We limit the number of sweep evaluations to ensure we have budget for refinement
    # e.g., 100 evaluations for sweep, 400 for refinement
    MAX_SWEEP_EVALS = 100
    
    sweep_count = 0
    bracket_a = None
    bracket_b = None
    bracket_fa = None
    bracket_fb = None
    
    # We need to store all points to check for sign changes between adjacent points
    # Sort by x later? Or just check adjacent in expansion order?
    # Adjacent in expansion order is not spatially adjacent.
    # Better: maintain a sorted list of points? Or just check all pairs?
    # With ~100 points, checking all pairs is O(N^2) = 10k ops, which is fine.
    
    # Actually, just checking adjacent in the *spatial* domain is best.
    # Let's collect points and sort them.
    
    all_points = [(best_x, best_fx)]
    
    while sweep_count < MAX_SWEEP_EVALS and check_time():
        # Expand left
        new_left_x = left_x - current_scale
        try:
            fx_left = call_f(new_left_x)
            sweep_count += 1
            all_points.append((new_left_x, fx_left))
            if math.isfinite(fx_left) and abs(fx_left) < abs(best_fx):
                best_x = new_left_x
                best_fx = fx_left
            left_x = new_left_x
        except Exception:
            pass
            
        if not check_time() or sweep_count >= MAX_SWEEP_EVALS:
            break
            
        # Expand right
        new_right_x = right_x + current_scale
        try:
            fx_right = call_f(new_right_x)
            sweep_count += 1
            all_points.append((new_right_x, fx_right))
            if math.isfinite(fx_right) and abs(fx_right) < abs(best_fx):
                best_x = new_right_x
                best_fx = fx_right
            right_x = new_right_x
        except Exception:
            pass
            
        # Increase scale
        current_scale *= r
        
        if current_scale > max_scale:
            break

    # Sort points by x to find adjacent sign changes
    all_points.sort(key=lambda p: p[0])
    
    # Check for sign change
    for i in range(len(all_points) - 1):
        x1, f1 = all_points[i]
        x2, f2 = all_points[i+1]
        
        if not (math.isfinite(f1) and math.isfinite(f2)):
            continue
            
        if f1 == 0:
            return x1
        if f2 == 0:
            return x2
            
        if f1 * f2 < 0:
            bracket_a = x1
            bracket_b = x2
            bracket_fa = f1
            bracket_fb = f2
            break

    # Phase 2: Refinement
    # If we have a bracket, use Bisection/Secant hybrid (Brent's method logic simplified)
    # If not, use the best point and try local Secant/Newton
    
    if bracket_a is not None:
        a, b = bracket_a, bracket_b
        fa, fb = bracket_fa, bracket_fb
        
        # Safeguarded Secant/Bisection
        # We want to converge fast. Secant is faster but can diverge.
        # Strategy: Try Secant step. If it leaves the bracket or doesn't reduce interval size, use Bisection.
        
        # Initialize secant points
        # We have a, b. We need a third point for secant?
        # Or just use the two points.
        
        # Let's use a modified Illinois/Brent approach.
        # Keep a, b, and c (previous b).
        c = a
        fc = fa
        
        # Ensure |fa| < |fb| to maintain invariant for Illinois
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            c = a
            fc = fa
            
        # Now |fa| <= |fb|
        
        while check_time() and eval_count < MAX_EVALS:
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
                
            # Bisection step
            x_mid = (a + b) / 2.0
            f_mid = None
            
            # Try Secant step
            # Secant formula: x_s = b - fb * (b - a) / (fb - fa)
            # This is the root of the line through (a, fa) and (b, fb)
            if abs(fb - fa) > 1e-15:
                x_secant = b - fb * (b - a) / (fb - fa)
                
                # Check if secant step is within bracket
                if a < x_secant < b:
                    try:
                        f_secant = call_f(x_secant)
                    except Exception:
                        f_secant = float('inf')
                    
                    if math.isfinite(f_secant):
                        # Accept secant step if it reduces the interval or is close
                        # Standard safeguard: if |f_secant| < |f_mid| (estimated) or just accept if in bracket
                        # Let's use a simple rule: accept secant if it's in bracket.
                        # Update bracket
                        if fa * f_secant < 0:
                            # Root in [a, x_secant]
                            b = x_secant
                            fb = f_secant
                        else:
                            # Root in [x_secant, b]
                            a = x_secant
                            fa = f_secant
                            
                        # Maintain invariant |fa| <= |fb|
                        if abs(fa) > abs(fb):
                            a, b = b, a
                            fa, fb = fb, fa
                            c = a
                            fc = fa
                            continue
                        else:
                            c = b
                            fc = fb
                            continue
                    else:
                        # Non-finite, fall back to bisection
                        pass
                        
            # Bisection Step
            try:
                f_mid = call_f(x_mid)
            except Exception:
                break
                
            if not math.isfinite(f_mid):
                # If mid is bad, shrink bracket?
                # Just break or try smaller step?
                # For robustness, break and return best
                break
                
            if abs(f_mid) <= 1e-8:
                return x_mid
                
            if fa * f_mid < 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
                
            # Maintain invariant |fa| <= |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                c = a
                fc = fa
            else:
                c = b
                fc = fb
                
            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(a)):
                break
                
    else:
        # No bracket found. Use best point and try local Secant/Newton
        # We have best_x, best_fx
        # Try to estimate derivative using a small step
        h = 1e-6 * max(1.0, abs(best_x))
        
        # Try secant with best_x and best_x + h
        x1 = best_x
        f1 = best_fx
        
        try:
            x2 = x1 + h
            f2 = call_f(x2)
        except Exception:
            x2 = x1 - h
            try:
                f2 = call_f(x2)
            except Exception:
                return best_x
                
        if not math.isfinite(f2):
            return best_x
            
        # Secant iteration
        for _ in range(50):
            if not check_time() or eval_count >= MAX_EVALS:
                break
                
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
                
            if abs(f1 - f2) < 1e-15:
                break
                
            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
            
            try:
                f_new = call_f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            # Update points
            # Keep the two best points? Or just shift?
            # Standard secant: (x1, f1) <- (x2, f2), (x2, f2) <- (x_new, f_new)
            # But if it diverges, we lose the good point.
            # Better: keep best point fixed?
            # Let's use a safeguard: if |f_new| > min(|f1|, |f2|), we might be diverging.
            
            if abs(f_new) < abs(best_fx):
                best_x = x_new
                best_fx = f_new
                
            # Simple shift
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            
            # If it's getting worse, stop
            if abs(f2) > 10 * abs(best_fx):
                break

    return best_x
# EVOLVE-BLOCK-END
