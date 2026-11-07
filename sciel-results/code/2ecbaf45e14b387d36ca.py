"""Root finder for 3 fixed smooth polynomial equations using Chebyshev probing and Brent's method."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # 2. Chebyshev Node Probing to find a bracket
    # Polynomials are well-behaved. We probe a range around x0 using Chebyshev nodes.
    # This is more robust than geometric expansion for finding the root location.
    
    # Determine an initial range. We'll expand if necessary, but start with a reasonable guess.
    # Since x0 is a starting point, it's likely near the root or at least in a relevant domain.
    # However, for polynomials, the root could be anywhere. 
    # Strategy: Try a sequence of ranges: [x0-1, x0+1], [x0-10, x0+10], etc.
    # But to save calls, we use Chebyshev nodes within the range to find sign changes or minima.
    
    # Let's try to find a bracket [a, b] such that f(a)*f(b) < 0 or f(a)==0 or f(b)==0.
    # If no sign change is found, we look for the minimum |f| in the range.
    
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    a_bracket = None
    b_bracket = None
    fa_bracket = None
    fb_bracket = None
    
    # Ranges to probe
    # We use a small number of calls for this phase.
    # Chebyshev nodes for N=5 in [-1, 1]:
    # cos(k*pi/5) for k=0..4
    # k=0: cos(0) = 1
    # k=1: cos(pi/5) = 0.809
    # k=2: cos(2pi/5) = 0.309
    # k=3: cos(3pi/5) = -0.309
    # k=4: cos(4pi/5) = -0.809
    # We also want the center 0.
    # So nodes: -0.809, -0.309, 0, 0.309, 0.809 (scaled to range)
    
    cheb_nodes = [-0.809016994, -0.309016994, 0.0, 0.309016994, 0.809016994]
    
    # We will try ranges of size 1, 10, 100, 1000 centered at x0
    # But we must be careful with time and calls.
    # Let's try to find a bracket in the first few ranges.
    
    found_bracket = False
    # Store points evaluated in this phase to potentially use for interpolation if no bracket
    probe_points = [(x, fx)]
    
    for scale in [1.0, 10.0, 100.0, 1000.0, 1e6]:
        if time.time() > deadline:
            break
            
        # Range is [x0 - scale, x0 + scale]
        # But if x0 is large, scale should be relative?
        # For polynomials, roots are often O(1) or related to coefficients.
        # Let's stick to absolute scale for now, assuming x0 is a decent hint.
        
        a = x0 - scale
        b = x0 + scale
        
        # Evaluate Chebyshev nodes in this range
        # Map t in [-1, 1] to x in [a, b]
        # x = (a + b)/2 + (b - a)/2 * t
        mid = (a + b) / 2.0
        half_range = (b - a) / 2.0
        
        # Evaluate nodes
        # We already have x0 (which is mid if scale is symmetric)
        # Let's evaluate the other 4 nodes
        
        for t in cheb_nodes:
            if time.time() > deadline:
                break
                
            if abs(t) < 1e-9:
                # This is x0, already evaluated
                continue
                
            x_node = mid + half_range * t
            try:
                f_node = f(x_node)
            except Exception:
                continue
                
            if not math.isfinite(f_node):
                continue
                
            # Update best
            if abs(f_node) < best_abs_fx:
                best_x = x_node
                best_fx = f_node
                best_abs_fx = abs(f_node)
                
            if abs(f_node) <= 1e-8:
                return x_node
                
            # Check for sign change with previous points
            # We can just check against x0 for simplicity in this phase, 
            # or check against all probe points.
            # Let's check against x0 first, then against each other.
            
            # Check with x0
            if fx * f_node < 0:
                # Bracket found!
                if abs(fx) < abs(f_node):
                    a_bracket, b_bracket = x0, x_node
                    fa_bracket, fb_bracket = fx, f_node
                else:
                    a_bracket, b_bracket = x_node, x0
                    fa_bracket, fb_bracket = f_node, fx
                found_bracket = True
                break
            
            # Check with other probe points (simple O(N^2) since N is small)
            for px, pf in probe_points:
                if px == x0:
                    continue # Already checked
                if pf * f_node < 0:
                    if abs(pf) < abs(f_node):
                        a_bracket, b_bracket = px, x_node
                        fa_bracket, fb_bracket = pf, f_node
                    else:
                        a_bracket, b_bracket = x_node, px
                        fa_bracket, fb_bracket = f_node, pf
                    found_bracket = True
                    break
            
            probe_points.append((x_node, f_node))
            
            if found_bracket:
                break
                
        if found_bracket:
            break
            
        # If we finished the loop for this scale without finding a bracket,
        # we might need to expand. But we also update best_x.
        
    if not found_bracket:
        # No bracket found. 
        # If we have a good best_x, we can try to refine it using Newton/secant locally.
        # Or just return best_x if it's close enough.
        if best_abs_fx <= 1e-8:
            return best_x
            
        # Try to use the best point and a neighbor to do a secant step
        # But we don't have a guaranteed neighbor.
        # Let's try to find a local sign change around best_x with small steps.
        # This is a fallback.
        
        # If we still don't have a bracket, we return the best point found.
        # This might fail the |f(x)| <= 1e-8 check, but it's the best we can do.
        # However, the task says "returns x with |f(x)| <= 1e-8".
        # If we can't find it, we fail.
        
        # Let's try one more thing: if we have 2 points, do secant.
        if len(probe_points) >= 2:
            # Sort by x
            probe_points.sort(key=lambda p: p[0])
            # Try secant on the two points with smallest |f|
            # Or just try secant on the last two
            x1, f1 = probe_points[-2]
            x2, f2 = probe_points[-1]
            if abs(f1 - f2) > 1e-30:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                try:
                    f_new = f(x_new)
                    if math.isfinite(f_new):
                        if abs(f_new) < best_abs_fx:
                            best_x = x_new
                            best_fx = f_new
                            best_abs_fx = abs(f_new)
                            if abs(f_new) <= 1e-8:
                                return best_x
                except Exception:
                    pass

        return best_x

    # 3. Brent's Method
    # We have a bracket [a_bracket, b_bracket] with fa_bracket, fb_bracket.
    # fa_bracket * fb_bracket < 0
    
    # Standard Brent's implementation
    a = a_bracket
    b = b_bracket
    c = a  # c is the previous value of b
    
    fa = fa_bracket
    fb = fb_bracket
    fc = fa
    
    # Ensure |b - x0| <= |a - x0| is not strictly required for Brent, 
    # but we usually maintain |b-a| < |c-b| or similar invariants.
    # Standard invariant: |b-a| < |a-c| and |f(b)| < |f(a)|
    
    # Let's initialize properly
    if abs(fb) < abs(fa):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    d = b - a  # Previous step size
    e = d     # Last interpolation step size
    
    # Convergence criteria
    tol = 1e-12
    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        if abs(fb) <= 1e-8:
            return b
            
        # Check for convergence
        if abs(b - a) < tol:
            return b
            
        # Invariant: |b-a| <= |a-c| and |f(b)| < |f(a)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            c = a
            fc = fa
            d = b - a
            e = d
            
        # Check for linear convergence
        if abs(e) < tol:
            return b
            
        # Attempt to construct a quadratic
        s = fa / fc
        q = 1.0
        r = 1.0
        
        if a == c:
            # Linear interpolation
            s = fb / fa
            q = 1.0
            r = 1.0
            # This part of Brent is tricky to implement perfectly from scratch.
            # Let's use a simplified safeguarded secant/IQI hybrid which is robust.
            # Actually, let's just use the standard algorithm carefully.
            
            # If a == c, we use secant
            # x_new = b - fb * (b - a) / (fb - fa)
            # This is the secant step.
            
            # Calculate secant step
            if abs(fa - fb) > 1e-30:
                x_secant = b - fb * (b - a) / (fb - fa)
            else:
                x_secant = b + 0.5 * (b - a)
                
            # Check if secant step is within [a, b]
            # If not, use bisection
            if x_secant > a and x_secant < b:
                x_new = x_secant
            else:
                x_new = 0.5 * (a + b)
                
        else:
            # Quadratic interpolation (IQI)
            # We have points (a, fa), (b, fb), (c, fc)
            # Fit x = A f^2 + B f + C
            # Or use the standard formula for x_new
            
            # Let's use the formula from Brent's paper or a standard reference.
            # x_new = b - fb * ( (b-a) * (fb-fc)/(fa-fc) - (b-c) * (fb-fa)/(fa-fa) ) / ( (b-a) * (fa-fb)/(fa-fc) - (b-c) * (fa-fb)/(fa-fa) ) ? No, that's messy.
            
            # Simpler: Inverse Quadratic Interpolation
            # x = A f^2 + B f + C
            # Lagrange:
            # x_new = a * fb*fc / ((fa-fb)*(fa-fc)) + b * fa*fc / ((fb-fa)*(fb-fc)) + c * fa*fb / ((fc-fa)*(fc-fb))
            
            denom1 = (fa - fb) * (fa - fc)
            denom2 = (fb - fa) * (fb - fc)
            denom3 = (fc - fa) * (fc - fb)
            
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                term1 = a * fb * fc / denom1
                term2 = b * fa * fc / denom2
                term3 = c * fa * fb / denom3
                x_iqi = term1 + term2 + term3
            else:
                x_iqi = None
                
            # Secant step
            if abs(fa - fb) > 1e-30:
                x_secant = b - fb * (b - a) / (fb - fa)
            else:
                x_secant = None
                
            # Bisection step
            x_bisect = 0.5 * (a + b)
            
            # Choose step
            x_new = None
            
            # Conditions for accepting IQI:
            # 1. x_iqi is within [a, b]
            # 2. x_iqi is closer to b than the last step e
            # 3. |x_iqi - b| < 0.5 * |b - c|
            
            if x_iqi is not None and a < x_iqi < b:
                if abs(x_iqi - b) < abs(e):
                    if abs(x_iqi - b) < 0.5 * abs(b - c):
                        x_new = x_iqi
                        
            # Conditions for accepting Secant:
            # 1. x_secant is within [a, b]
            # 2. x_secant is closer to b than the last step e
            
            if x_new is None and x_secant is not None and a < x_secant < b:
                if abs(x_secant - b) < abs(e):
                    x_new = x_secant
                    
            # Otherwise, bisection
            if x_new is None:
                x_new = x_bisect
                
        # Evaluate f(x_new)
        try:
            fx_new = f(x_new)
        except Exception:
            # If evaluation fails, fall back to bisection
            x_new = 0.5 * (a + b)
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break
                
        if not math.isfinite(fx_new):
            break
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Update variables
        d = e
        e = b - x_new
        
        if fa * fx_new < 0:
            a = x_new
            fa = fx_new
        else:
            c = b
            fc = fb
            b = x_new
            fb = fx_new
            
        # Maintain invariant: |f(b)| < |f(a)|
        if abs(fb) < abs(fa):
            # Swap a and b if needed? No, Brent's method handles this by swapping a,b,c.
            # Standard Brent swaps a and b if |fa| < |fb| at the start of the loop.
            # We did that at the top of the loop.
            pass

    # Return the best point found
    if abs(fb) <= abs(fa):
        return b
    else:
        return a

# EVOLVE-BLOCK-END
