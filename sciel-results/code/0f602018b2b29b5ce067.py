"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _safe_f(f, x):
    try:
        y = f(x)
        if not math.isfinite(y):
            return float('inf')
        return float(y)
    except Exception:
        return float('inf')

def _hybrid_secant_bisection(f, x0, max_iter=30):
    """
    Hybrid root finder using Secant method with Bisection fallback.
    
    Strategy:
    1. Initialize two points x0 and x1 = x0 + eps.
    2. Iterate:
       a. Calculate Secant estimate x_new.
       b. Check if x_new is "safe" (within the bracket defined by x0, x1 or close to them).
       c. If safe and |f(x_new)| is decreasing, accept x_new.
       d. If unsafe or not decreasing, use a Bisection-like step to stay in the "good" region.
    3. Stop if |f(x)| <= 1e-10 or max_iter reached.
    """
    x0 = float(x0)
    # Small initial step to get a second point
    # Scale step based on x0 magnitude
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    
    fx0 = _safe_f(f, x0)
    fx1 = _safe_f(f, x1)
    
    # If x0 is already a root
    if abs(fx0) <= 1e-12:
        return x0
    if abs(fx1) <= 1e-12:
        return x1

    # If we have a sign change, we can use a more robust bracketing approach
    # But Secant is often faster if we stay close.
    # Let's maintain a bracket [a, b] if possible, otherwise just track last 2 points.
    
    # To ensure robustness, let's try to find a bracket first with minimal cost?
    # No, the directive says "different design". Let's stick to pure hybrid secant/bisection
    # but we need a fallback if we diverge.
    
    # We will track the best x found so far
    best_x = x0 if abs(fx0) < abs(fx1) else x1
    best_fx = min(abs(fx0), abs(fx1))
    
    # If no sign change, Secant might diverge. 
    # A safer "Hybrid" is often: Try Secant. If it goes outside [min(x0,x1), max(x0,x1)] * factor, 
    # or if |f(x_new)| > max(|f(x0)|, |f(x1)|), then bisect.
    
    # However, without a bracket, bisection is hard. 
    # Let's use a "Trust Region" approach.
    # Center = best_x. Radius = distance between last two points.
    # If Secant step is within [Center - 2*Radius, Center + 2*Radius], try it.
    # Else, move towards the point with smaller |f| by half the distance (bisection-like).
    
    for i in range(max_iter):
        # Determine current points
        # We keep x_prev (older) and x_curr (newer)
        # Let's swap so x_curr is the one with smaller |f|? No, Secant needs distinct points.
        
        # Secant formula: x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        denom = fx1 - fx0
        if abs(denom) < 1e-15:
            # Derivative approx 0, or points too close.
            # Move x1 slightly
            x1 = x0 + 1e-8 * max(1.0, abs(x0))
            fx1 = _safe_f(f, x1)
            if abs(fx1) <= 1e-12: return x1
            continue

        x_new = x1 - fx1 * (x1 - x0) / denom
        
        if not math.isfinite(x_new):
            # Diverged to inf. Fallback: move towards the better point
            if abs(fx0) < abs(fx1):
                x_new = (x0 + x1) / 2.0
            else:
                x_new = (x0 + x1) / 2.0
            # Actually, if Secant explodes, bisection is safer.
            # Let's just take the midpoint of the last two points.
            x_new = (x0 + x1) / 2.0
        
        fx_new = _safe_f(f, x_new)
        
        if not math.isfinite(fx_new):
            # Bad point. Shrink step.
            # Move halfway to the better of x0, x1
            if abs(fx0) <= abs(fx1):
                x_new = (x0 + x1) / 2.0
            else:
                x_new = (x0 + x1) / 2.0
            fx_new = _safe_f(f, x_new)
            if not math.isfinite(fx_new):
                # If even midpoint is bad, we are stuck. Return best.
                break

        # Check if this is a good step
        # Condition 1: |f(x_new)| < min(|f(x0)|, |f(x1)|)
        # Condition 2: x_new is not too far (Trust Region)
        # Let's define trust region as the interval spanned by x0 and x1, expanded by 50%.
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        span = max_x - min_x
        if span == 0:
            span = 1e-8
        
        # If x_new is within [min_x - 0.5*span, max_x + 0.5*span] AND improves f, accept.
        # Otherwise, use a bisection-like step:
        # If sign change exists between x0 and x1, bisect.
        # If no sign change, move towards the point with smaller |f| by a factor (e.g., 0.5)
        
        if abs(fx_new) < best_fx:
            best_x = x_new
            best_fx = abs(fx_new)
            
        if abs(fx_new) <= 1e-10:
            return x_new

        # Decide whether to accept x_new as the new x1 or use a fallback
        # Accept if it improves upon the worst of the two current points
        # AND is within a reasonable distance
        dist_new = abs(x_new - x1)
        
        # Simple heuristic:
        # If f(x_new) < max(f(x0), f(x1)) and x_new is finite:
        #   If sign change between x0 and x_new: x1 = x_new, x0 stays (or swap to keep order)
        #   If sign change between x1 and x_new: x0 = x_new, x1 stays
        #   Else (no sign change): 
        #       If |f(x_new)| < |f(x0)|: x0 = x_new
        #       If |f(x_new)| < |f(x1)|: x1 = x_new
        #       Else: Fallback to bisection/move towards best
        
        # To simplify and ensure progress:
        # If x_new is "good" (reduces max residual), use it.
        # Else, force a move towards the best known point.
        
        max_res = max(abs(fx0), abs(fx1))
        
        if abs(fx_new) < max_res and math.isfinite(x_new):
            # Update points
            # We want to keep the two points that define the "best" local behavior.
            # If x_new is better than x0, replace x0? 
            # If x_new is better than x1, replace x1?
            # Let's replace the worse one.
            if abs(fx0) > abs(fx1):
                # x0 is worse. Replace x0 with x_new? 
                # But Secant needs distinct points.
                # If x_new is close to x1, we might lose curvature info.
                # Let's just shift the window: x0 = x1, x1 = x_new
                # This is essentially a "step" in the direction of x_new.
                x0, fx0 = x1, fx1
                x1, fx1 = x_new, fx_new
            else:
                # x1 is worse (or equal). 
                # If we replace x1, we lose the "current" position.
                # Better to shift window: x0 = x1, x1 = x_new is standard for Secant iteration.
                # Let's always shift the window forward if the step is accepted.
                x0, fx0 = x1, fx1
                x1, fx1 = x_new, fx_new
        else:
            # Rejection. Use a safe move.
            # Move x1 halfway to x0 (if x0 is better) or x0 halfway to x1 (if x1 is better)
            # This acts like bisection on the interval [x0, x1]
            if abs(fx0) < abs(fx1):
                # x0 is better. Move x1 towards x0.
                x1 = (x0 + x1) / 2.0
                fx1 = _safe_f(f, x1)
            else:
                # x1 is better. Move x0 towards x1.
                x0 = (x0 + x1) / 2.0
                fx0 = _safe_f(f, x0)
            
            if abs(fx1) < best_fx:
                best_x = x1
                best_fx = abs(fx1)
            if abs(fx0) < best_fx:
                best_x = x0
                best_fx = abs(fx0)
                
            if abs(fx1) <= 1e-10: return x1
            if abs(fx0) <= 1e-10: return x0

    return best_x

def find_root(f, x0):
    """
    Adaptive local search with global fallback.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    # Phase 1: Local Hybrid Secant/Bisection
    # This should converge for smooth polynomials if x0 is reasonably close.
    x_local = _hybrid_secant_bisection(f, x0, max_iter=20)
    fx_local = _safe_f(f, x_local)
    
    if math.isfinite(fx_local) and abs(fx_local) <= 1e-8:
        return x_local

    # Check time
    if time.time() > deadline - 5.0:
        return x_local

    # Phase 2: Global Search if local failed
    # If local failed, the root is likely far from x0.
    # We need to find a bracket.
    # Strategy: Exponential expansion from x0 to find a sign change.
    
    a = x0
    fa = _safe_f(f, a)
    if not math.isfinite(fa):
        # If x0 is invalid, try 0.0
        a = 0.0
        fa = _safe_f(f, a)
        if not math.isfinite(fa):
            return 0.0

    if abs(fa) <= 1e-8:
        return a

    # Try expanding
    scale = 1.0
    found_bracket = False
    best_x = a
    best_fx = abs(fa)
    
    # Also try the local result as a potential candidate
    if abs(fx_local) < best_fx:
        best_x = x_local
        best_fx = abs(fx_local)

    while time.time() < deadline - 10.0:
        # Try right and left
        for direction in [1.0, -1.0]:
            b = x0 + direction * scale
            fb = _safe_f(f, b)
            
            if not math.isfinite(fb):
                continue

            if abs(fb) < best_fx:
                best_x = b
                best_fx = abs(fb)
                
            if fa * fb < 0:
                # Found bracket [a, b]
                found_bracket = True
                break
            
            # Update a to b if we are expanding in one direction? 
            # No, we are expanding from x0. 
            # If no sign change, we just keep expanding.
            # But we should check if we crossed a root without sign change? 
            # For polynomials, roots are isolated. If f is smooth, sign change is reliable 
            # unless the root is even multiplicity (tangent). 
            # If tangent, f(a) and f(b) have same sign. 
            # In that case, we might miss it. 
            # However, the task says "smooth polynomial equations". 
            # Usually implies simple roots or we can detect local minima of |f|.
            
            # For now, rely on sign change. If we expand too much, we might miss.
            # Let's also check if |f| is decreasing towards 0 without sign change.
            # If fb < fa and fb < fb_prev, we might be approaching a tangent root.
            # This is complex. Let's stick to sign change for robustness.
            
            # Optimization: If we haven't found a bracket, we should update 'a' to the 
            # point with smallest |f| to help with the "best_x" tracking, but for bracketing,
            # we need to keep the original x0 or the last point that didn't cross?
            # Actually, if we expand from x0, and f(x0) is positive, and f(x0+10) is positive,
            # we don't know if there was a root in between.
            # To be safe, we should step in smaller increments if we suspect a root.
            # But exponential expansion is standard for "find a bracket".
            
            # Let's refine: if we don't find a bracket, we continue.
            # But we should also check intermediate points if the scale is large?
            # No, that increases calls.
            
            # Let's just break out of the direction loop if we found a bracket.
            if found_bracket:
                break
        
        if found_bracket:
            break
        
        scale *= 2.0
        if scale > 1e10:
            break

    if found_bracket:
        # Refine with Illinois/Newton hybrid (reuse logic or simple bisection)
        # Use a simple bisection to be safe and fast enough
        # Or use the local hybrid again on the bracket?
        # Let's use a simple bisection to guarantee convergence within budget.
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            m = (a + b) / 2.0
            fm = _safe_f(f, m)
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            if abs(b - a) < 1e-12:
                break
        return (a + b) / 2.0

    # If no bracket found, return the best point found
    return best_x

# EVOLVE-BLOCK-END
