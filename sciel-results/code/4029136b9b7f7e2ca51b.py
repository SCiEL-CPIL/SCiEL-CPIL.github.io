import math
import time
import random

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a global search strategy.
    1. Quick local Newton check (cheap, 2 calls)
    2. Global random search to find a bracket (sign change)
    3. Illinois method to converge (fast, ~15-20 calls)
    """
    # Deadline guard: 145s out of 150s
    deadline = time.time() + 145.0
    tol = 1e-8  # Target tolerance
    x = float(x0)
    
    # Best solution found so far
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f)
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(x_val) or f_val is None:
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
            
    # Check x0
    update_best(x, best_f)
    if best_abs <= tol:
        return best_x

    # Phase 1: Very fast local Newton (only 2 calls: f(x) and f(x+h))
    # This catches cases where x0 is already very close to a root
    h = 1e-6 * max(1.0, abs(x))
    x_newton = x + h
    fx_newton = f(x_newton)
    update_best(x_newton, fx_newton)
    
    if abs(fx_newton) <= tol:
        return x_newton
    
    # If derivative is non-zero, try one Newton step
    d = (fx_newton - best_f) / h
    if d != 0 and math.isfinite(d):
        x_n = x - best_f / d
        if math.isfinite(x_n):
            fx_n = f(x_n)
            update_best(x_n, fx_n)
            if abs(fx_n) <= tol:
                return x_n
            # If this new point is much better, use it as the new center for search
            if abs(fx_n) < best_abs:
                x = x_n
                best_f = fx_n
                best_abs = abs(fx_n)
                # Try one more Newton step from here
                h2 = 1e-6 * max(1.0, abs(x))
                x_n2 = x + h2
                fx_n2 = f(x_n2)
                update_best(x_n2, fx_n2)
                if abs(fx_n2) <= tol:
                    return x_n2
                d2 = (fx_n2 - best_f) / h2
                if d2 != 0 and math.isfinite(d2):
                    x_n3 = x - best_f / d2
                    if math.isfinite(x_n3):
                        fx_n3 = f(x_n3)
                        update_best(x_n3, fx_n3)
                        if abs(fx_n3) <= tol:
                            return x_n3

    # Phase 2: Global Search for a Bracket
    # Since functions are oscillatory, roots are likely close to x0.
    # We sample points in a logarithmic grid around x0 to find a sign change.
    
    # Define search ranges
    # We want to find [a, b] such that f(a)*f(b) < 0.
    # We'll try a series of intervals.
    
    # Strategy: 
    # 1. Try small intervals around x0
    # 2. If no sign change, try larger intervals
    # 3. If still no sign change, try random sampling
    
    # We already have f(x0). Let's store it.
    # Note: best_f might have been updated to a different point. 
    # But for bracketing, we need consistent endpoints.
    # Let's restart the bracketing search with fresh evaluations if needed, 
    # or use the existing ones if they are valid.
    
    # To minimize calls, we'll construct a set of candidate points and evaluate them.
    # We'll prioritize points close to x0.
    
    # Generate candidate points
    candidates = []
    # Small steps
    for step in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        candidates.append(x0 + step)
        candidates.append(x0 - step)
        
    # Shuffle to add some randomness? No, deterministic is better for reproducibility 
    # and we want to cover the space systematically first.
    # However, "oscillatory" might mean high frequency. 
    # Let's add some random points too.
    random.seed(42) # For reproducibility in evaluation
    for _ in range(20):
        # Log-uniform distribution around x0
        # Range: 1e-5 to 1e3 relative to x0 scale
        scale = max(1.0, abs(x0))
        offset = random.uniform(1e-5, 1e3) * scale
        sign = 1 if random.random() < 0.5 else -1
        candidates.append(x0 + sign * offset)
        
    # Sort candidates by distance to x0 to prioritize local search
    candidates.sort(key=lambda c: abs(c - x0))
    
    # We have f(x0) stored in best_f? Not necessarily, if Newton improved it.
    # Let's get f(x0) specifically if not already known.
    # Actually, we can just start evaluating candidates.
    # We need a pair (a, b) with sign change.
    
    # Store evaluated points
    # point -> f(point)
    evals = {}
    
    # Add x0 to evals
    evals[x0] = best_f if abs(best_f) == abs(f(x0)) else f(x0) 
    # Wait, best_f might be from x_n. Let's just re-evaluate x0 to be safe and simple?
    # No, that wastes a call. 
    # Let's assume x0 is a good anchor. 
    # If best_x != x0, we might have found a root already. 
    # If not, we proceed.
    
    # Let's use a list of points we have values for.
    # Start with x0.
    points = [(x0, best_f)]
    
    # If best_x is different from x0, we have another point.
    if abs(best_x - x0) > 1e-12:
        # We need f(best_x). We have best_f.
        # But best_f corresponds to best_x.
        points.append((best_x, best_f))
        
    # Function to check for sign change among known points
    def check_bracket(points_list):
        # Sort by x
        sorted_pts = sorted(points_list, key=lambda p: p[0])
        for i in range(len(sorted_pts) - 1):
            x1, f1 = sorted_pts[i]
            x2, f2 = sorted_pts[i+1]
            if f1 * f2 < 0:
                return (x1, f1, x2, f2)
            # Check if any is a root
            if abs(f1) <= tol:
                return (x1, f1, x1, f1) # Degenerate bracket
            if abs(f2) <= tol:
                return (x2, f2, x2, f2)
        return None
        
    bracket = check_bracket(points)
    if bracket:
        a, fa, b, fb = bracket
        if a == b:
            return a
    else:
        # Evaluate candidates until we find a bracket or run out of time/budget
        # Limit to 30 more evaluations for search phase
        search_limit = 30
        for c in candidates:
            if time.time() > deadline:
                break
            if len(points) > search_limit:
                break
            # Skip if too close to existing points
            if any(abs(c - p[0]) < 1e-12 for p in points):
                continue
            fc = f(c)
            update_best(c, fc)
            if abs(fc) <= tol:
                return c
            points.append((c, fc))
            bracket = check_bracket(points)
            if bracket:
                a, fa, b, fb = bracket
                if a == b:
                    return a
                break
                
    if bracket is None:
        # No bracket found. Return best found so far.
        # This is a failure case for the root finding requirement, but we return best_x.
        return best_x

    # Phase 3: Illinois Method
    # Converges faster than bisection for many functions.
    # Max 50 iterations to be safe.
    for _ in range(50):
        if time.time() > deadline:
            break
        mid = (a + b) / 2.0
        fm = f(mid)
        update_best(mid, fm)
        if abs(fm) <= tol:
            return mid
        
        if fa * fm < 0:
            b = mid
            fb = fm
            # Illinois: shrink a
            a = (a + b) / 2.0
            fa = f(a) # This is an extra call! 
            # Standard Illinois replaces the endpoint with the midpoint.
            # Let's stick to standard Illinois:
            # If f(a)*f(m) < 0: b=m, fb=fm, a=(a+b)/2, fa=f(a)
            # This requires 2 calls per iteration (mid and new a).
            # Bisection requires 1 call per iteration.
            # Illinois converges in ~1.5x fewer iterations but 2x calls?
            # Actually, Illinois reduces interval by factor of 2/3 (approx) per step vs 1/2 for bisection?
            # No, Illinois is faster. Let's use a simpler variant or just Bisection to save calls?
            # Bisection: 1 call/iter. 1e-8 precision needs ~27 iterations (2^27 ~ 1e8).
            # Illinois: ~20 iterations.
            # If Illinois takes 2 calls/iter, that's 40 calls.
            # Bisection takes 27 calls.
            # So Bisection is more call-efficient for high precision!
            # Let's use Bisection.
            
            # Reset to Bisection logic
            break # Exit Illinois loop, switch to Bisection below
            
        else:
            a = mid
            fa = fm
            # Illinois: shrink b
            b = (a + b) / 2.0
            fb = f(b)
            
    # If we broke out of Illinois early (due to sign change check logic above being flawed for pure bisection)
    # Let's just implement pure Bisection here for reliability and low call count.
    
    # We have a bracket [a, b] with fa, fb.
    # Ensure fa * fb < 0
    if fa * fb > 0:
        # Should not happen if bracket was valid, but check just in case
        # If one is zero, return it
        if abs(fa) <= tol: return a
        if abs(fb) <= tol: return b
        # Otherwise, we are stuck. Return best.
        return best_x

    # Pure Bisection
    for _ in range(40): # 2^40 is plenty for double precision
        if time.time() > deadline:
            break
        mid = (a + b) / 2.0
        fm = f(mid)
        update_best(mid, fm)
        if abs(fm) <= tol:
            return mid
        
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
            
        # Check if interval is small enough
        if abs(b - a) < 1e-12:
            break
            
    return best_x
# EVOLVE-BLOCK-END
