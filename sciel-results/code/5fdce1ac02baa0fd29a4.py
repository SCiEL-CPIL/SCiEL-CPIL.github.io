"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder using Inverse Quadratic Interpolation (IQI) with safeguards.
    Strategy:
    1. Collect 3 initial points near x0
    2. Use IQI to find next iterate (more stable than Newton for ill-conditioned f)
    3. Fallback to bisection if IQI step is too large or out of bracket
    4. Adaptive step damping to handle flat regions
    5. Deadline guard to stay within time limits
    """
    deadline = time.time() + 145  # Leave 5s margin before 150s kill
    max_evals = 490
    eval_count = [0]
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')
    
    def eval_f(x):
        """Evaluate f with counting and deadline check."""
        nonlocal best_x, best_fx, best_abs_fx
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if val is None or math.isnan(val) or math.isinf(val):
                return None
            abs_val = abs(val)
            if abs_val < best_abs_fx:
                best_abs_fx = abs_val
                best_x = x
                best_fx = val
            return val
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            return None
    
    def is_good(x, fx):
        return fx is not None and not math.isnan(fx) and not math.isinf(fx) and abs(fx) <= 1e-8
    
    # Phase 1: Quick check at x0
    fx0 = eval_f(float(x0))
    if fx0 is not None and is_good(x0, fx0):
        return x0
    
    # Phase 2: Collect 3 points for IQI
    # Start with x0, then two nearby points
    x1 = float(x0)
    h = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h
    x3 = x1 - h
    
    fx1 = fx0 if fx0 is not None else None
    fx2 = eval_f(x2)
    fx3 = eval_f(x3)
    
    # If any point is already good, return it
    for x, fx in [(x1, fx1), (x2, fx2), (x3, fx3)]:
        if fx is not None and is_good(x, fx):
            return x
    
    # Build list of valid (x, f(x)) pairs
    points = []
    for x, fx in [(x1, fx1), (x2, fx2), (x3, fx3)]:
        if fx is not None and not math.isnan(fx) and not math.isinf(fx):
            points.append((x, fx))
    
    # If we have fewer than 2 valid points, try expanding
    if len(points) < 2:
        for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            h_exp = scale * max(1.0, abs(x1))
            for direction in [1, -1]:
                x_try = x1 + direction * h_exp
                fx_try = eval_f(x_try)
                if fx_try is not None and not math.isnan(fx_try) and not math.isinf(fx_try):
                    points.append((x_try, fx_try))
                    if is_good(x_try, fx_try):
                        return x_try
                    if len(points) >= 3:
                        break
            if len(points) >= 3:
                break
    
    if len(points) < 2:
        return best_x
    
    # Sort points by x for bracketing logic
    points.sort(key=lambda p: p[0])
    
    # Check if we have a sign change (bracket)
    a, b = None, None
    fa, fb = None, None
    for i in range(len(points) - 1):
        x_a, f_a = points[i]
        x_b, f_b = points[i + 1]
        if f_a * f_b < 0:
            a, b = x_a, x_b
            fa, fb = f_a, f_b
            break
    
    # Phase 3: Main iteration loop using IQI with safeguards
    # Maintain three most recent points
    if len(points) >= 3:
        # Use the three closest points to x0
        points_sorted_by_dist = sorted(points, key=lambda p: abs(p[0] - x1))
        x_a, f_a = points_sorted_by_dist[0]
        x_b, f_b = points_sorted_by_dist[1]
        x_c, f_c = points_sorted_by_dist[2]
    elif len(points) == 2:
        x_a, f_a = points[0]
        x_b, f_b = points[1]
        x_c, f_c = None, None
    else:
        return best_x
    
    # Main iteration
    for iteration in range(200):
        if time.time() > deadline:
            break
        if eval_count[0] >= max_evals - 10:
            break
        
        # Check if any current point is good
        for x, fx in [(x_a, f_a), (x_b, f_b), (x_c, f_c)]:
            if fx is not None and is_good(x, fx):
                return x
        
        # Determine next iterate
        x_new = None
        f_new = None
        
        # Strategy 1: IQI if we have 3 distinct points with distinct f values
        if x_c is not None and x_a != x_b and x_b != x_c and x_a != x_c:
            if f_a != f_b and f_b != f_c and f_a != f_c:
                # Inverse quadratic interpolation
                # Fit parabola through (f_a, x_a), (f_b, x_b), (f_c, x_c)
                # Find where parabola crosses f=0
                # Using Lagrange interpolation in inverse form:
                # x(f) = x_a * L_a(f) + x_b * L_b(f) + x_c * L_c(f)
                # where L_a(f) = (f - f_b)(f - f_c) / ((f_a - f_b)(f_a - f_c))
                # etc.
                denom_a = (f_a - f_b) * (f_a - f_c)
                denom_b = (f_b - f_a) * (f_b - f_c)
                denom_c = (f_c - f_a) * (f_c - f_b)
                
                if abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30:
                    L_a = (0 - f_b) * (0 - f_c) / denom_a
                    L_b = (0 - f_a) * (0 - f_c) / denom_b
                    L_c = (0 - f_a) * (0 - f_b) / denom_c
                    x_iq = x_a * L_a + x_b * L_b + x_c * L_c
                    
                    if not math.isnan(x_iq) and not math.isinf(x_iq):
                        # Safeguard: if we have a bracket, keep within it
                        if a is not None and b is not None:
                            lo, hi = min(a, b), max(a, b)
                            if lo < x_iq < hi:
                                x_new = x_iq
                            else:
                                # IQI gave out-of-bracket result, use bisection
                                x_new = (a + b) / 2
                        else:
                            # No bracket, use IQI if step is reasonable
                            ref_x = x_b  # Use middle point as reference
                            if abs(x_iq - ref_x) < 10 * max(1.0, abs(ref_x)):
                                x_new = x_iq
                            else:
                                # Step too large, use secant instead
                                x_new = None
        
        # Strategy 2: Secant if IQI didn't give a good step
        if x_new is None and x_a != x_b and f_a != f_b:
            x_sec = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            if not math.isnan(x_sec) and not math.isinf(x_sec):
                # Safeguard
                if a is not None and b is not None:
                    lo, hi = min(a, b), max(a, b)
                    if lo < x_sec < hi:
                        x_new = x_sec
                    else:
                        x_new = (a + b) / 2
                else:
                    ref_x = x_b
                    if abs(x_sec - ref_x) < 10 * max(1.0, abs(ref_x)):
                        x_new = x_sec
                    else:
                        x_new = None
        
        # Strategy 3: Bisection if we have a bracket
        if x_new is None and a is not None and b is not None:
            x_new = (a + b) / 2
        
        # Strategy 4: Adaptive step from best point
        if x_new is None:
            # Use the point with smallest |f| and take a small step
            candidates = [(x_a, f_a), (x_b, f_b)]
            if x_c is not None and f_c is not None:
                candidates.append((x_c, f_c))
            candidates = [c for c in candidates if c[1] is not None and not math.isnan(c[1]) and not math.isinf(c[1])]
            if candidates:
                best_cand = min(candidates, key=lambda c: abs(c[1]))
                x_ref, f_ref = best_cand
                step = 1e-3 * max(1.0, abs(x_ref))
                # Choose direction based on sign of f
                x_new = x_ref + step if f_ref > 0 else x_ref - step
            else:
                break
        
        # Evaluate new point
        f_new = eval_f(x_new)
        if f_new is None:
            break
        if is_good(x_new, f_new):
            return x_new
        
        # Update bracket if possible
        if a is not None and b is not None and fa is not None and fb is not None:
            if fa * f_new <= 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
        
        # Update the three-point set: keep the three most "useful" points
        # Prefer points that are close together and have distinct f values
        all_points = [(x_a, f_a), (x_b, f_b), (x_new, f_new)]
        if x_c is not None and f_c is not None:
            all_points.append((x_c, f_c))
        
        # Select 3 points: prefer those that form a good bracket or are close
        # Simple heuristic: keep the two closest to x_new and one that's distinct
        all_points.sort(key=lambda p: abs(p[0] - x_new))
        
        # Ensure we keep diversity
        x_a, f_a = all_points[0]
        x_b, f_b = all_points[1] if len(all_points) > 1 else (x_a, f_a)
        x_c, f_c = all_points[2] if len(all_points) > 2 else (None, None)
        
        # If we lost the bracket, try to re-establish
        if a is None or b is None:
            for i in range(len(all_points) - 1):
                if all_points[i][1] * all_points[i+1][1] < 0:
                    a, b = all_points[i][0], all_points[i+1][0]
                    fa, fb = all_points[i][1], all_points[i+1][1]
                    break
    
    # Final polish: if we're close, do a few secant iterations
    if best_abs_fx < 1e-5:
        x_polish = best_x
        fx_polish = best_fx
        x_prev = x_polish + 1e-8 * max(1.0, abs(x_polish))
        fx_prev = eval_f(x_prev)
        
        for i in range(30):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            if fx_prev is None or math.isnan(fx_prev) or math.isinf(fx_prev):
                break
            if fx_polish == fx_prev:
                break
            x_new = x_polish - fx_polish * (x_polish - x_prev) / (fx_polish - fx_prev)
            if math.isnan(x_new) or math.isinf(x_new):
                break
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if is_good(x_new, fx_new):
                return x_new
            x_prev, fx_prev = x_polish, fx_polish
            x_polish, fx_polish = x_new, fx_new
    
    return best_x
# EVOLVE-BLOCK-END
