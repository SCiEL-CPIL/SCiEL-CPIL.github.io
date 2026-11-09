"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _illinois_newton_hybrid(f, a, b, fa, fb, max_iter=30, tol=1e-12):
    """
    Hybrid root finder.
    1. If fa and fb have opposite signs, use Illinois method (modified bisection) 
       to robustly narrow the bracket.
    2. Once the bracket is small enough, switch to Newton's method with 
       finite difference derivative for quadratic convergence.
    3. If no sign change is detected initially, it returns the point with 
       smallest |f(x)| found during a brief local scan, but this function 
       is primarily called when a sign change is expected or found by the caller.
    """
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # If no sign change, we can't bracket. 
    # However, the caller (find_root) usually ensures a sign change or handles 
    # the "no bracket" case by returning the best point found so far.
    # For this hybrid, we assume a sign change exists. If not, we do a few 
    # bisection-like steps to minimize |f|, but really we should just return best.
    if fa * fb > 0:
        # No bracket. Just return the midpoint or the better of a, b.
        # But wait, the caller might have found a local min. 
        # We'll just return the point with smaller |f| for safety, 
        # though find_root will handle the "no bracket" logic.
        return a if abs(fa) < abs(fb) else b

    x = (a + b) / 2.0
    fx = f(x)
    if abs(fx) <= tol:
        return x

    # Illinois Method Phase: Robust bracketing
    # We keep the bracket [a, b] such that f(a)*f(b) < 0.
    # We shrink the side that is "further" from the root in a weighted way.
    
    # To switch to Newton, we need a good estimate of the derivative.
    # We can estimate it from the secant of the current bracket or use FD.
    # Let's run Illinois for a few steps to get close, then Newton.
    
    # Actually, a simpler robust approach:
    # 1. Do 5-10 bisection/Illinois steps to get the interval small.
    # 2. Use Newton from the midpoint with FD derivative.
    
    # Illinois step logic:
    # If f(a)*f(x) < 0, root is in [a, x]. 
    # Weight: if f(a) is "smaller" in magnitude than f(b), we trust the side 
    # with smaller f more? No, Illinois weights based on function values to 
    # avoid shrinking the side with the larger function value too slowly.
    
    # Let's implement a standard Illinois step for 10 iterations.
    for _ in range(10):
        if not math.isfinite(fx):
            # If we hit a non-finite value, fall back to bisection
            x = (a + b) / 2.0
            fx = f(x)
            if abs(fx) <= tol:
                return x
            if not math.isfinite(fx):
                break
            if fa * fx < 0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
            continue
            
        if abs(fx) <= tol:
            return x

        if fa * fx < 0:
            # Root in [a, x]
            # Illinois: weight the interval based on |fa| and |fb|
            # Standard Illinois: new_x = (a*|fb| + x*|fa|) / (|fa| + |fb|)
            # But we want to shrink the side with the LARGER function value.
            # If |fa| < |fb|, the root is likely closer to a.
            # So we shrink [x, b] more aggressively? No, we shrink [a, x] less?
            # Actually, Illinois replaces the endpoint with the larger function value.
            # If |fa| > |fb|, replace a with a weighted point.
            # If |fa| < |fb|, replace b with a weighted point.
            
            # Let's stick to simple bisection for robustness in the first 5 steps,
            # then switch to Newton. Bisection is reliable.
            b = x
            fb = fx
        else:
            a = x
            fa = fx
            
        # Check if bracket is small enough for Newton
        if abs(b - a) < 1e-8 * max(1.0, abs(x)):
            break

    # Newton Phase
    # Start from the midpoint of the current bracket
    x = (a + b) / 2.0
    fx = f(x)
    
    for _ in range(15):
        if abs(fx) <= tol:
            return x
        if not math.isfinite(fx):
            break
            
        h = 1e-7 * max(1.0, abs(x))
        fxp = f(x + h)
        fxm = f(x - h)
        
        if not math.isfinite(fxp) or not math.isfinite(fxm):
            # Fall back to bisection if FD fails
            # Just do a few bisection steps
            for _ in range(5):
                x = (a + b) / 2.0
                fx = f(x)
                if abs(fx) <= tol:
                    return x
                if fa * fx < 0:
                    b = x
                    fb = fx
                else:
                    a = x
                    fa = fx
            break
            
        if fxp == fxm:
            break
            
        d = (fxp - fxm) / (2 * h)
        if d == 0 or not math.isfinite(d):
            break
            
        x_new = x - fx / d
        
        # Damping: if x_new is outside the bracket, move to the midpoint of the bracket and the new point
        if (x_new - a) * (x_new - b) > 0:
            # x_new is outside [a, b]
            # Move towards the root but stay within or close to the bracket
            # Simple damping: x_new = (x + x_new) / 2? Or just bisect?
            # Let's just bisect to be safe if Newton steps away.
            x_new = (a + b) / 2.0
            
        x = x_new
        fx = f(x)
        
        # Update bracket for safety
        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
            
    return x

def find_root(f, x0):
    """
    Adaptive Multi-Start Root Finder.
    1. Evaluate f(x0).
    2. If |f(x0)| is small, return x0.
    3. Expand a bracket around x0 by testing x0 +/- delta * 10^k.
    4. If a sign change is found, use the Hybrid Illinois-Newton method.
    5. If no sign change is found in a reasonable range, return the point 
       with the minimum |f(x)| observed.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    x0 = float(x0)
    fx0 = f(x0)
    
    if not math.isfinite(fx0):
        # If x0 is bad, we have to search.
        # Try a few nearby points
        best_x = x0
        best_f = float('inf')
        for dx in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            x_try = x0 + dx
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) < best_f:
                best_f = abs(f_try)
                best_x = x_try
            x_try = x0 - dx
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) < best_f:
                best_f = abs(f_try)
                best_x = x_try
        return best_x

    if abs(fx0) <= 1e-8:
        return x0

    # We need to find a bracket [a, b] such that f(a)*f(b) < 0.
    # We will expand from x0.
    # We keep track of the best point (smallest |f|) seen so far.
    best_x = x0
    best_abs_f = abs(fx0)
    
    # Scales to try: 1, 10, 100, ... and 0.1, 0.01, ...
    # We alternate left and right to find the root quickly if it's close.
    # Strategy: 
    # 1. Try small steps to catch nearby roots.
    # 2. Expand exponentially.
    
    # Let's maintain a list of "anchors" to check for sign changes.
    # Anchor 0 is x0.
    # We check x0 + d, x0 - d, x0 + 10d, x0 - 10d, etc.
    
    # To minimize calls, we only evaluate a point if it might close a bracket.
    # We keep the current "frontier" on left and right.
    # Left frontier: x_left, f_left. Initially x0, fx0.
    # Right frontier: x_right, f_right. Initially x0, fx0.
    
    x_left = x0
    f_left = fx0
    x_right = x0
    f_right = fx0
    
    # We will try scales: 1e-6, 1e-5, ..., 1e6
    # But we don't want to do 14 scales * 2 sides = 28 calls if we can avoid it.
    # We stop as soon as we find a sign change.
    
    scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7]
    
    # We also need to handle the case where the root is very close and we step over it?
    # No, if we step over it, f(x0) and f(x0+d) will have opposite signs.
    
    # We should also check if the function is monotonic and we just need to minimize |f|.
    # If no sign change is found, we return the best_x.
    
    found_bracket = False
    a = 0.0
    b = 0.0
    fa = 0.0
    fb = 0.0
    
    for scale in scales:
        if time.time() > deadline:
            break
            
        # Check right side
        x_try_r = x0 + scale
        if math.isfinite(x_try_r):
            f_try_r = f(x_try_r)
            if math.isfinite(f_try_r):
                if abs(f_try_r) < best_abs_f:
                    best_abs_f = abs(f_try_r)
                    best_x = x_try_r
                    
                # Check for sign change with x_right
                if f_right * f_try_r < 0:
                    a = x_right
                    fa = f_right
                    b = x_try_r
                    fb = f_try_r
                    found_bracket = True
                    break
                else:
                    # Update frontier if f_try_r is "smaller" in magnitude?
                    # Or just keep expanding? 
                    # If f_right and f_try_r have same sign, the root is not between them.
                    # We can update x_right to x_try_r if we want to keep the frontier far,
                    # but if f_try_r is much larger, maybe we should keep the closer one?
                    # For bracketing, it's safer to keep the one with smaller |f| 
                    # if signs are same? No, if signs are same, we just expand.
                    # However, if we have f(x0)=1, f(x0+1)=-1, we found it.
                    # If f(x0)=1, f(x0+1)=2, we update x_right to x0+1?
                    # If the next step is x0+10, and f(x0+10)=-1, we have bracket [x0+1, x0+10].
                    # This is fine.
                    x_right = x_try_r
                    f_right = f_try_r
        
        # Check left side
        x_try_l = x0 - scale
        if math.isfinite(x_try_l):
            f_try_l = f(x_try_l)
            if math.isfinite(f_try_l):
                if abs(f_try_l) < best_abs_f:
                    best_abs_f = abs(f_try_l)
                    best_x = x_try_l
                    
                if f_left * f_try_l < 0:
                    a = x_try_l
                    fa = f_try_l
                    b = x_left
                    fb = f_left
                    found_bracket = True
                    break
                else:
                    x_left = x_try_l
                    f_left = f_try_l
    
    if found_bracket:
        # We have a bracket [a, b] with a sign change.
        # Use the hybrid solver.
        root = _illinois_newton_hybrid(f, a, b, fa, fb)
        # Verify the root
        f_root = f(root)
        if math.isfinite(f_root) and abs(f_root) < best_abs_f:
            return root
        else:
            # If the hybrid solver failed or returned something worse, 
            # return the best point found during bracketing.
            return best_x
    else:
        # No bracket found. Return the point with smallest |f(x)|.
        return best_x

# EVOLVE-BLOCK-END
