"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting from x0 using inverse quadratic interpolation
    with bisection fallback for robustness on ill-conditioned functions.
    
    Returns x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    """
    MAX_EVALS = 500
    TOL = 1e-8
    
    eval_count = 0
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise ValueError("Evaluation limit exceeded")
        eval_count += 1
        return f(x)
    
    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= TOL:
        return x
    
    # Try to establish a bracket by probing in both directions with expanding steps
    a, b, fa, fb = None, None, None, None
    base_step = 1e-3 * max(1.0, abs(x))
    
    # More aggressive bracketing: try larger initial steps and more scales
    for direction in [1, -1]:
        for scale in [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
            if eval_count >= MAX_EVALS:
                break
            x_try = x + direction * base_step * scale
            if x_try == x:
                continue
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if fx * fx_try < 0:
                if fx < fx_try:
                    a, b = x, x_try
                    fa, fb = fx, fx_try
                else:
                    a, b = x_try, x
                    fa, fb = fx_try, fx
                break
        if a is not None:
            break
    
    # If we found a bracket, use IQI with bisection fallback
    if a is not None:
        # History for IQI: maintain last 3 distinct points
        history = [(a, fa), (b, fb)]
        
        for _ in range(MAX_EVALS - eval_count):
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
            new_x = None
            
            # Try IQI if we have 3 distinct points with distinct f values
            if len(history) >= 3:
                x1, f1 = history[-3]
                x2, f2 = history[-2]
                x3, f3 = history[-1]
                
                # Check for distinct f values (needed for IQI)
                if abs(f1 - f2) > 1e-12 and abs(f2 - f3) > 1e-12 and abs(f1 - f3) > 1e-12:
                    # Use Lagrange interpolation to find root of quadratic
                    # P(x) = f1 * L1(x) + f2 * L2(x) + f3 * L3(x)
                    # where Li are Lagrange basis polynomials
                    # Set P(x) = 0 and solve for x
                    
                    # Coefficients of the quadratic in standard form: ax^2 + bx + c = 0
                    # From Lagrange:
                    # P(x) = f1 * (x-x2)(x-x3)/((x1-x2)(x1-x3)) 
                    #      + f2 * (x-x1)(x-x3)/((x2-x1)(x2-x3))
                    #      + f3 * (x-x1)(x-x2)/((x3-x1)(x3-x2))
                    
                    d1 = (x1 - x2) * (x1 - x3)
                    d2 = (x2 - x1) * (x2 - x3)
                    d3 = (x3 - x1) * (x3 - x2)
                    
                    if abs(d1) > 1e-30 and abs(d2) > 1e-30 and abs(d3) > 1e-30:
                        # Expand to get quadratic coefficients
                        # Term1: f1/d1 * (x^2 - (x2+x3)x + x2*x3)
                        # Term2: f2/d2 * (x^2 - (x1+x3)x + x1*x3)
                        # Term3: f3/d3 * (x^2 - (x1+x2)x + x1*x2)
                        
                        c2 = f1/d1 + f2/d2 + f3/d3
                        c1 = -f1*(x2+x3)/d1 - f2*(x1+x3)/d2 - f3*(x1+x2)/d3
                        c0 = f1*x2*x3/d1 + f2*x1*x3/d2 + f3*x1*x2/d3
                        
                        # Solve c2*x^2 + c1*x + c0 = 0
                        if abs(c2) > 1e-30:
                            disc = c1*c1 - 4*c2*c0
                            if disc >= 0:
                                sqrt_disc = disc ** 0.5
                                root1 = (-c1 + sqrt_disc) / (2*c2)
                                root2 = (-c1 - sqrt_disc) / (2*c2)
                                
                                # Pick the root within the bracket
                                lo, hi = min(a, b), max(a, b)
                                margin = (hi - lo) * 1e-8
                                for root in [root1, root2]:
                                    if lo + margin < root < hi - margin:
                                        new_x = root
                                        break
                        elif abs(c1) > 1e-30:
                            # Linear case
                            root = -c0 / c1
                            lo, hi = min(a, b), max(a, b)
                            margin = (hi - lo) * 1e-8
                            if lo + margin < root < hi - margin:
                                new_x = root
            
            if new_x is None:
                # Fallback: use the point closer to the root (based on function value)
                # If fa and fb have opposite signs, bisect
                # Otherwise, move toward the point with smaller |f|
                if fa * fb < 0:
                    new_x = (a + b) / 2.0
                else:
                    # No sign change; move toward the endpoint with smaller |f|
                    if abs(fa) < abs(fb):
                        new_x = a + 0.5 * (b - a)
                    else:
                        new_x = b - 0.5 * (a - b)
            
            fx_new = safe_f(new_x)
            if abs(fx_new) <= TOL:
                return new_x
            
            # Update bracket: keep the sign-change interval
            if fa * fx_new < 0:
                b, fb = new_x, fx_new
            elif fb * fx_new < 0:
                a, fa = new_x, fx_new
            else:
                # No sign change with new point; keep the better bracket
                if abs(fx_new) < abs(fa):
                    a, fa = new_x, fx_new
                elif abs(fx_new) < abs(fb):
                    b, fb = new_x, fx_new
                else:
                    # Keep existing bracket
                    pass
            
            # Update history: keep last 3 distinct points
            history.append((new_x, fx_new))
            # Remove duplicates and keep only last 3
            unique = []
            for pt in history:
                if not any(abs(pt[0] - u[0]) < 1e-14 for u in unique):
                    unique.append(pt)
            history = unique[-3:]
        
        return a if abs(fa) < abs(fb) else b
    
    # No bracket found: use secant method with adaptive stepping
    x1, f1 = x, fx
    x2, f2 = None, None
    
    # Try a few different step sizes to find a good second point
    best_x2, best_f2 = None, None
    for step_mult in [1e-3, 1e-2, 0.1, 1.0]:
        if eval_count >= MAX_EVALS:
            break
        step = step_mult * max(1.0, abs(x))
        x_try = x + step
        if x_try == x:
            continue
        f_try = safe_f(x_try)
        if abs(f_try) <= TOL:
            return x_try
        # Prefer points with smaller |f|
        if best_f2 is None or abs(f_try) < abs(best_f2):
            best_x2, best_f2 = x_try, f_try
    
    if best_x2 is not None:
        x2, f2 = best_x2, best_f2
    
    if x2 is None:
        # Last resort: small step
        step = 1e-6 * max(1.0, abs(x))
        x2 = x + step
        f2 = safe_f(x2)
        if abs(f2) <= TOL:
            return x2
    
    for _ in range(MAX_EVALS - eval_count):
        if abs(f1) <= TOL:
            return x1
        if abs(f2) <= TOL:
            return x2
        
        # Secant step
        if abs(f2 - f1) > 1e-12:
            new_x = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            # Nearly parallel; take a step toward the point with smaller |f|
            if abs(f1) < abs(f2):
                new_x = x1 + 0.5 * (x2 - x1)
            else:
                new_x = x2 + 0.5 * (x1 - x2)
        
        fx_new = safe_f(new_x)
        if abs(fx_new) <= TOL:
            return new_x
        
        # Keep the two points with smaller |f|
        if abs(fx_new) < abs(f1):
            x1, f1 = x2, f2
            x2, f2 = new_x, fx_new
        elif abs(fx_new) < abs(f2):
            x2, f2 = new_x, fx_new
        else:
            x1, f1 = x2, f2
            x2, f2 = new_x, fx_new
    
    return x1 if abs(f1) < abs(f2) else x2

# EVOLVE-BLOCK-END
