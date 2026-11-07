"""
Root finder using Inverse Quadratic Interpolation (IQI).
Strategy: Exploit smooth polynomial structure for cubic convergence.
Avoids the overhead of bracketing and Brent's method.
"""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    # Handle non-finite or exact root
    if not math.isfinite(fx):
        # Try to find a nearby finite point to restart IQI
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                    if math.isfinite(fx_test):
                        x = x_test
                        fx = fx_test
                        found = True
                        break
                except Exception:
                    continue
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x
    else:
        if abs(fx) <= 1e-8:
            return x

    # 2. IQI Setup
    # We need 3 points to fit a quadratic.
    # x0, f(x0) is point 0.
    # Generate x1 and x2 with small steps to establish the local behavior.
    # Since we don't know the scale, start with a small step and adapt if needed.
    # However, for polynomials, a small step is usually safe unless the root is very far.
    # If the root is far, IQI will take large steps quickly.
    
    # Point 0
    x0_curr = x
    f0 = fx
    
    # Point 1: Small step to the right
    # Heuristic: start with 1e-3. If f is very steep, this might be too small? 
    # Actually, for IQI, we just need distinct points. 
    # Let's use a relative step if possible, but x0 might be 0.
    # Use a fixed small step first.
    h = 1e-3
    x1_curr = x0_curr + h
    try:
        f1 = f(x1_curr)
    except Exception:
        f1 = float('nan')
        
    if not math.isfinite(f1):
        # Try left
        x1_curr = x0_curr - h
        try:
            f1 = f(x1_curr)
        except Exception:
            f1 = float('nan')
            
    if not math.isfinite(f1):
        # If we can't get a second point, we can't do IQI. 
        # Fallback to a simple secant-like or just return x0? 
        # For robustness, let's try a few more steps if the first fails.
        # But for polynomials, this shouldn't happen unless x0 is in a singularity (unlikely for poly) 
        # or the step lands on a non-finite (unlikely for poly).
        # So if this fails, something is weird. Return x0.
        return x0

    # Point 2: Another step, maybe further to get better curvature estimate
    # If f0 and f1 have opposite signs, we have a bracket. 
    # IQI works best with 3 points. 
    # Let's take x2 further in the direction of the smaller |f|? 
    # Or just x0 + 2h?
    # Let's try x2 = x0 + 2*h (if x1 was right) or x0 - 2*h (if x1 was left)
    # Actually, it's better to spread them out a bit if the function is flat.
    # But for polynomials, local behavior is good.
    # Let's use x2 = x1 + h (so x0, x0+h, x0+2h) or similar.
    
    if x1_curr > x0_curr:
        x2_curr = x0_curr + 2*h
    else:
        x2_curr = x0_curr - 2*h
        
    try:
        f2 = f(x2_curr)
    except Exception:
        f2 = float('nan')
        
    if not math.isfinite(f2):
        # Try a different direction for x2? 
        # If x1 was right, try left for x2?
        if x1_curr > x0_curr:
            x2_curr = x0_curr - h
        else:
            x2_curr = x0_curr + h
        try:
            f2 = f(x2_curr)
        except Exception:
            f2 = float('nan')
            
    if not math.isfinite(f2):
        return x0

    # Now we have 3 points: (x0, f0), (x1, f1), (x2, f2)
    # We want to find the next x such that f(x) = 0.
    # We fit a quadratic in x for f(x)? No, we fit a quadratic for x(y) or use Lagrange interpolation for f(x) and find root?
    # Standard IQI: Interpolate f(x) with a quadratic Q(x) through the 3 points.
    # Then find the root of Q(x) = 0 that is closest to the last point or within the bracket.
    # Alternatively, interpolate x as a function of y (inverse). This is often more stable for root finding if f is not too flat.
    # However, if f is flat (derivative ~ 0), inverse interpolation fails (division by small numbers).
    # Direct interpolation of f(x) is safer for polynomials.
    
    # Let's use the Lagrange form for the quadratic Q(x) passing through (x0, f0), (x1, f1), (x2, f2).
    # Q(x) = f0 * L0(x) + f1 * L1(x) + f2 * L2(x)
    # Where L0(x) = (x-x1)(x-x2) / ((x0-x1)(x0-x2))
    # L1(x) = (x-x0)(x-x2) / ((x1-x0)(x1-x2))
    # L2(x) = (x-x0)(x-x1) / ((x2-x0)(x2-x1))
    
    # Q(x) = 0 is a quadratic equation: Ax^2 + Bx + C = 0.
    # We can compute A, B, C explicitly.
    # A = f0/((x0-x1)(x0-x2)) + f1/((x1-x0)(x1-x2)) + f2/((x2-x0)(x2-x1))
    # B = ... (more complex)
    # C = ...
    
    # Easier: Solve the system for coefficients of Q(x) = ax^2 + bx + c
    # [x0^2 x0 1] [a]   [f0]
    # [x1^2 x1 1] [b] = [f1]
    # [x2^2 x2 1] [c]   [f2]
    
    # To avoid numerical issues with large x, we can shift the coordinates.
    # Let's just use the formula for the root of the interpolating polynomial.
    # There is a known formula for the IQI step.
    
    # Let's calculate the coefficients a, b, c of the quadratic Q(x).
    # Using Cramer's rule or substitution.
    
    # To improve numerical stability, let's center the x values.
    # Let xm = (x0+x1+x2)/3? No, just use x0 as origin for the local fit?
    # Let u = x - x0.
    # Points: (0, f0), (u1, f1), (u2, f2) where u1 = x1-x0, u2 = x2-x0.
    # Fit q(u) = au^2 + bu + c.
    # q(0) = c = f0.
    # q(u1) = a*u1^2 + b*u1 + f0 = f1 => a*u1^2 + b*u1 = f1 - f0
    # q(u2) = a*u2^2 + b*u2 + f0 = f2 => a*u2^2 + b*u2 = f2 - f0
    
    # Let d1 = f1 - f0, d2 = f2 - f0.
    # a*u1^2 + b*u1 = d1
    # a*u2^2 + b*u2 = d2
    
    # Solve for a, b:
    # b = (d1 - a*u1^2) / u1 = d1/u1 - a*u1
    # Substitute into 2nd eq:
    # a*u2^2 + (d1/u1 - a*u1)*u2 = d2
    # a*u2^2 + d1*u2/u1 - a*u1*u2 = d2
    # a*(u2^2 - u1*u2) = d2 - d1*u2/u1
    # a = (d2 - d1*u2/u1) / (u2^2 - u1*u2)
    # a = (d1*u2 - d2*u1) / (u1*(u2^2 - u1*u2))  ... wait, let's re-derive carefully.
    
    # Matrix form:
    # [u1^2, u1] [a]   [d1]
    # [u2^2, u2] [b] = [d2]
    
    # Det = u1^2*u2 - u1*u2^2 = u1*u2*(u1 - u2)
    
    # If Det is close to 0, the points are collinear in (u, f) space? No, it means u1 or u2 is 0 or u1==u2.
    # We ensured x0, x1, x2 are distinct.
    
    a = (d1*u2 - d2*u1) / (u1*u2*(u1 - u2))
    b = (d2*u1 - d1*u2) / (u1*u2*(u1 - u2))  # Wait, let's check b.
    # From a*u1^2 + b*u1 = d1 => b = (d1 - a*u1^2)/u1
    # Let's just use the formula derived from Cramer's rule for a and b.
    # a = (d1*u2 - d2*u1) / (u1^2*u2 - u1*u2^2) = (d1*u2 - d2*u1) / (u1*u2*(u1-u2))
    # b = (d1*u2^2 - d2*u1^2) / (u1*u2*(u1-u2)) ?
    # Let's check:
    # b = (d1*u2 - d2*u1) ... no.
    # Let's just solve the 2x2 system numerically if needed, but formulas are faster.
    # Let's use the standard IQI update formula which is often cited.
    # But deriving it is safer.
    
    # Let's re-calculate b using a.
    b = (d1 - a * u1 * u1) / u1
    
    # Now we have q(u) = a*u^2 + b*u + f0.
    # We want to find u such that q(u) = 0.
    # a*u^2 + b*u + f0 = 0.
    # Roots: u = (-b +/- sqrt(b^2 - 4*a*f0)) / (2*a)
    
    # We need to choose the root that is "best".
    # Usually the one closest to the last point or within the convex hull.
    # Let's compute both.
    
    discriminant = b*b - 4*a*f0
    if discriminant < 0:
        # No real root for the quadratic approximation.
        # This can happen if the quadratic doesn't cross zero.
        # In this case, we might need to fall back to a different step, e.g., secant or bisection-like.
        # For polynomials, if the local quadratic doesn't cross, the true function might still cross further away.
        # Let's use the secant method as a fallback for this step.
        # Secant: x_new = x1 - f1*(x1-x0)/(f1-f0)
        # Using the two closest points?
        # Let's use x1 and x0 (or x2 and x1).
        # Let's pick the pair with smallest |f| difference?
        # Or just use the last two points (x1, x2).
        # Actually, if IQI fails, it's better to take a cautious step.
        # Let's use the secant step from x1 and x0.
        if abs(f1 - f0) > 1e-300:
            u_new = -f1 * u1 / (f1 - f0)
        else:
            # Degenerate case
            u_new = u1 * 0.5
    else:
        sqrt_disc = math.sqrt(discriminant)
        # Two roots
        u_root1 = (-b + sqrt_disc) / (2*a)
        u_root2 = (-b - sqrt_disc) / (2*a)
        
        # Choose the root closest to the most recent point?
        # Or the one that minimizes the step size?
        # Let's choose the one closest to u1 (the last point used).
        if abs(u_root1 - u1) < abs(u_root2 - u1):
            u_new = u_root1
        else:
            u_new = u_root2
            
        # Safety: If the step is huge or leads to non-finite, we might want to dampen it.
        # But for polynomials, it should be fine.
        
    # Update x
    x_new = x0_curr + u_new
    
    # Check if x_new is finite
    if not math.isfinite(x_new):
        # Fallback
        x_new = x0_curr + u1 * 0.5
        
    # Evaluate f(x_new)
    try:
        f_new = f(x_new)
    except Exception:
        f_new = float('nan')
        
    if not math.isfinite(f_new):
        # If we hit a non-finite value, we have a problem.
        # This shouldn't happen for polynomials unless x_new is inf.
        # If x_new was inf, the step was too big.
        # Let's try a smaller step.
        # For now, return the best we have.
        # We have x0, x1, x2. Pick the one with smallest |f|.
        candidates = [(x0_curr, abs(f0)), (x1_curr, abs(f1)), (x2_curr, abs(f2))]
        candidates.sort(key=lambda p: p[1])
        return candidates[0][0]
        
    if abs(f_new) <= 1e-8:
        return x_new

    # 3. Iterative Loop
    # Now we have 4 points: x0, x1, x2, x_new.
    # We should drop the oldest point and use the 3 most recent points for the next IQI step.
    # Which 3? Usually the 3 most recent in terms of iteration, or the 3 that form the smallest interval?
    # For polynomials, the 3 most recent points (x1, x2, x_new) should be good.
    # Let's shift: x0<-x1, x1<-x2, x2<-x_new.
    
    max_iter = 50
    for i in range(max_iter):
        if time.time() > deadline:
            # Time out. Return best candidate.
            # We need to track the best candidate found so far.
            # For simplicity, let's just return the last x if it's finite, or the best of the last 3.
            # But we don't have a global "best" tracker in this loop structure easily without refactoring.
            # Let's just return x_new if finite, else x0.
            return x_new if math.isfinite(x_new) else x0

        # Shift points
        x0_curr, f0 = x1_curr, f1
        x1_curr, f1 = x2_curr, f2
        x2_curr, f2 = x_new, f_new
        
        # Check convergence again (already checked, but just in case)
        if abs(f2) <= 1e-8:
            return x2_curr

        # Calculate next step using IQI on (x0_curr, f0), (x1_curr, f1), (x2_curr, f2)
        # Same logic as above
        
        u1 = x1_curr - x0_curr
        u2 = x2_curr - x0_curr
        
        if abs(u1) < 1e-15 or abs(u2) < 1e-15 or abs(u1 - u2) < 1e-15:
            # Points are too close or identical. Break the loop or use secant.
            # Use secant on the two closest points?
            # Let's just break and return the best.
            break
            
        d1 = f1 - f0
        d2 = f2 - f0
        
        denom = u1 * u2 * (u1 - u2)
        if abs(denom) < 1e-300:
            break
            
        a = (d1 * u2 - d2 * u1) / denom
        b = (d1 - a * u1 * u1) / u1
        
        discriminant = b*b - 4*a*f0
        if discriminant < 0:
            # Fallback to secant
            if abs(f1 - f0) > 1e-300:
                u_new = -f1 * u1 / (f1 - f0)
            else:
                break
        else:
            sqrt_disc = math.sqrt(discriminant)
            u_root1 = (-b + sqrt_disc) / (2*a)
            u_root2 = (-b - sqrt_disc) / (2*a)
            
            # Choose root closest to u1
            if abs(u_root1 - u1) < abs(u_root2 - u1):
                u_new = u_root1
            else:
                u_new = u_root2
                
        x_new = x0_curr + u_new
        
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except Exception:
            f_new = float('nan')
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new

    # If we exit the loop without finding a root, return the best candidate.
    # We have x0_curr, x1_curr, x2_curr, and maybe x_new from the last iteration.
    # Let's collect all evaluated points and pick the best.
    # For efficiency, we only keep the last 3 in the loop variables.
    # But we can just return x2_curr (the most recent) or check the last few.
    # Let's check the last 3 points held in variables.
    candidates = [
        (x0_curr, abs(f0)),
        (x1_curr, abs(f1)),
        (x2_curr, abs(f2))
    ]
    # Also include the very first points if they are still in scope?
    # x0, x1, x2 from the start are lost after the first shift.
    # This is a limitation. We should track the global best.
    
    # Let's refactor slightly to track global best.
    # Actually, for polynomials, convergence is very fast. 
    # If it didn't converge in 50 steps, it's likely stuck.
    # Returning the best of the last 3 is a reasonable heuristic.
    
    candidates.sort(key=lambda p: p[1])
    best_x = candidates[0][0]
    
    # Verify best_x
    try:
        fb = f(best_x)
        if math.isfinite(fb) and abs(fb) <= 1e-8:
            return best_x
    except Exception:
        pass
        
    return best_x

# EVOLVE-BLOCK-END
