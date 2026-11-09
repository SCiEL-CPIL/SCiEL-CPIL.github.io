"""Hybrid: efficient outward bracketing + Illinois-style refinement (1 call/step)."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0          # safety margin before 150 s hard kill
    TOL = 1e-12               # aim well below 1e-8
    HARD_TOL = 1e-8           # contract tolerance
    MAX_REFINE_ITERS = 400

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    # Evaluate start
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('nan')

    best_x = x0
    best_val = abs(f0) if (f0 == f0 and math.isfinite(f0)) else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx == fx and math.isfinite(fx):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    # If start is already good
    if f0 == f0 and math.isfinite(f0) and abs(f0) <= TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with a sign change, searching outward
    #    from x0 with an adaptive step that grows to catch the nearest
    #    root quickly, plus dense local sampling to catch oscillatory
    #    roots that change sign between coarse samples.
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    base = max(1.0, abs(x0))

    # First pass: coarse outward scan for a sign change (cheap, few calls).
    # We use a spiral: alternate forward and backward with growing steps.
    # Start with small steps to catch nearby roots, then grow.
    
    # We will maintain the last point in each direction.
    last_fwd_x, last_fwd_f = x0, f0
    last_bwd_x, last_bwd_f = x0, f0

    # Sequence of multipliers for the step size relative to base.
    # We want to start small and grow.
    # 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, ...
    # But since we are doing a spiral, we can just iterate k.
    # step = base * 10^(k - 4) ?
    # Let's use a fixed list of steps to be safe and deterministic.
    
    steps = [base * s for s in
             [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]]
    
    # To be more robust, let's interleave and also check the immediate neighbors.
    # Actually, the previous program's approach of a linear scan with growing step was decent.
    # Let's try a hybrid:
    # 1. Check x0 +/- base * 1e-4, 1e-3, ... 
    # 2. If not found, do a denser scan.

    found_bracket = False
    
    # Spiral search
    # We'll try steps in increasing order.
    # For each step, try forward then backward.
    for s in steps:
        if not elapsed_ok():
            break
            
        # Forward
        x_new = last_fwd_x + s
        try:
            f_new = f(x_new)
        except Exception:
            f_new = float('nan')
        track(x_new, f_new)
        
        if f_new == f_new and math.isfinite(f_new):
            if last_fwd_f == last_fwd_f and math.isfinite(last_fwd_f):
                if last_fwd_f * f_new < 0:
                    a, b, fa, fb = last_fwd_x, x_new, last_fwd_f, f_new
                    found_bracket = True
                    break
                if abs(f_new) <= TOL:
                    return x_new
            last_fwd_x, last_fwd_f = x_new, f_new
        else:
            # If NaN, we can't use it for bracketing, but we can continue?
            # Better to reset last_fwd to a known good point? Or just skip?
            # If f_new is NaN, we don't know the sign.
            # Let's just not update last_fwd_f if it's NaN, so we compare against the last good one?
            # No, that might miss a bracket if the function is undefined in between.
            # But for oscillatory functions, NaNs are unlikely unless domain errors.
            # Let's assume f is well-defined on the domain of interest.
            # If f_new is NaN, we break the chain of sign checks in this direction.
            pass

        if not elapsed_ok():
            break

        # Backward
        x_new = last_bwd_x - s
        try:
            f_new = f(x_new)
        except Exception:
            f_new = float('nan')
        track(x_new, f_new)
        
        if f_new == f_new and math.isfinite(f_new):
            if last_bwd_f == last_bwd_f and math.isfinite(last_bwd_f):
                if last_bwd_f * f_new < 0:
                    a, b, fa, fb = x_new, last_bwd_x, f_new, last_bwd_f
                    found_bracket = True
                    break
                if abs(f_new) <= TOL:
                    return x_new
            last_bwd_x, last_bwd_f = x_new, f_new
        else:
            pass

    # If coarse scan found nothing, do a denser local scan to catch
    # oscillatory sign changes that were skipped.
    if not found_bracket:
        # Dense scan over a few radii with moderate point counts.
        # We focus on the region where we haven't found a bracket yet.
        # The spiral covered specific points. Let's scan intervals between them?
        # Or just scan a wider range densely.
        
        # Let's scan [x0 - 10*base, x0 + 10*base] with N points.
        # But if base is large, 10*base is huge.
        # Let's try a few radii.
        
        radii = [base * 1e-2, base * 1e-1, base, base * 10.0]
        
        for r in radii:
            if not elapsed_ok():
                break
            # Number of sub-samples. 
            # For oscillatory functions, we need enough points to catch sign changes.
            # Let's use 1000 points.
            n = 1000
            left = x0 - r
            right = x0 + r
            prev_x = left
            try:
                prev_f = f(prev_x)
            except Exception:
                prev_f = float('nan')
            track(prev_x, prev_f)
            
            for i in range(1, n + 1):
                if not elapsed_ok():
                    break
                x_i = left + (right - left) * i / n
                try:
                    f_i = f(x_i)
                except Exception:
                    f_i = float('nan')
                track(x_i, f_i)
                
                if f_i == f_i and math.isfinite(f_i) and prev_f == prev_f and math.isfinite(prev_f):
                    if prev_f * f_i < 0:
                        a, b, fa, fb = prev_x, x_i, prev_f, f_i
                        found_bracket = True
                        break
                    if abs(f_i) <= TOL:
                        return x_i
                prev_x, prev_f = x_i, f_i
            
            if found_bracket:
                break

    # ------------------------------------------------------------------
    # 2) Refine the bracket with Illinois-style method (1 call/step).
    # ------------------------------------------------------------------
    if found_bracket and a is not None:
        # Ensure a < b for consistency in some implementations, but not strictly necessary for Illinois.
        # Let's just use the values as is.
        # Illinois method:
        # Keep (x, fx) and (y, fy).
        # If |fx| < |fy|, swap so that we always evaluate the midpoint of the longer interval?
        # No, Illinois always evaluates the midpoint of the interval, but swaps the endpoints 
        # if the new point is closer to the root than the old one on the same side?
        
        # Standard Illinois:
        # 1. Evaluate midpoint m.
        # 2. If sign(f(m)) == sign(f(x)), set y = m, fy = f(m).
        #    Else set x = m, fx = f(m).
        # 3. If |fx| < |fy|, swap (x, fx) and (y, fy).
        #    This ensures that the point with the smaller |f| is always 'x'.
        #    Wait, that's not quite right.
        #    Illinois: 
        #    - Evaluate m.
        #    - If f(m) has same sign as f(x), replace y with m.
        #    - Else replace x with m.
        #    - If |f(x)| < |f(y)|, swap x and y. 
        #      (This step ensures that the interval is shortened on the side with the larger |f|? No.)
        #      Actually, the swap ensures that the "current best" is always at x.
        #      And the next midpoint is taken as (x + y)/2?
        #      No, the key to Illinois is that if the new point is "better" (smaller |f|) than the 
        #      point it replaced, we swap the endpoints so that the "worse" point is on the side 
        #      that will be shortened next?
        
        # Let's implement a robust Illinois variant:
        # We maintain a bracket [lo, hi] with f(lo)*f(hi) < 0.
        # We also track which side has the smaller |f|.
        
        if fa is None or fb is None:
            try: fa = f(a)
            except: fa = float('nan')
            try: fb = f(b)
            except: fb = float('nan')
        
        # Initialize
        x1, f1 = a, fa
        x2, f2 = b, fb
        
        # Ensure f1 * f2 < 0
        if f1 * f2 > 0:
            # This shouldn't happen if we found a bracket, but check for safety
            # If not, we can't use Illinois. Fall back to bisection?
            pass

        for _ in range(MAX_REFINE_ITERS):
            if not elapsed_ok():
                break
                
            if f1 == f1 and math.isfinite(f1) and abs(f1) <= TOL:
                return x1
            if f2 == f2 and math.isfinite(f2) and abs(f2) <= TOL:
                return x2
            
            # Midpoint
            mid = 0.5 * (x1 + x2)
            
            try:
                fm = f(mid)
            except Exception:
                fm = float('nan')
            
            track(mid, fm)
            
            if fm == fm and math.isfinite(fm):
                if abs(fm) <= TOL:
                    return mid
                
                # Update bracket
                if f1 * fm < 0:
                    # Root is in [x1, mid]
                    x2, f2 = mid, fm
                else:
                    # Root is in [mid, x2]
                    x1, f1 = mid, fm
                
                # Illinois Swap:
                # If |f1| < |f2|, swap (x1, f1) and (x2, f2).
                # This effectively shortens the interval on the side with the larger |f| 
                # in the next step? 
                # Actually, if we swap, then in the next iteration, the midpoint is still the midpoint.
                # The "Illinois" acceleration comes from the fact that if one side has a much 
                # smaller |f|, we swap so that the "worse" point becomes the one that is 
                # replaced by the midpoint? 
                
                # Let's check the logic.
                # If we don't swap, we always bisect.
                # If we swap, we are effectively doing a secant-like step?
                
                # Standard Illinois:
                # After updating the bracket, if |f(x1)| < |f(x2)|, swap x1 and x2.
                # This means the point with the larger |f| is always at x1? No, smaller |f| is at x1.
                # Wait, if |f1| < |f2|, we swap. So now x1 is the one with larger |f|?
                # No, if |f1| < |f2|, and we swap, then new x1 is old x2 (larger |f|).
                # So x1 always holds the point with the LARGER |f|?
                
                # Let's re-read a reference.
                # "If |f(x1)| < |f(x2)|, interchange x1 and x2."
                # This ensures that |f(x1)| >= |f(x2)|.
                # Then the next midpoint is calculated.
                # Why does this help?
                # It turns out that this is equivalent to a secant method step in some cases?
                
                # Let's just implement it as described.
                if abs(f1) < abs(f2):
                    x1, f1, x2, f2 = x2, f2, x1, f1
            else:
                # NaN handling: shrink towards the side with finite value?
                # If fm is NaN, we can't use it. 
                # We might just break or try a smaller step?
                # For now, break to avoid infinite loops on NaNs.
                break

        # After refinement, return the best-seen point
        if best_val <= HARD_TOL:
            return best_x

    # ------------------------------------------------------------------
    # 3) No bracket found: do a short local Newton/Brent-style attempt
    #    from x0 as a last resort, then return the best finite candidate.
    # ------------------------------------------------------------------
    try:
        x = x0
        fx = f0
        for _ in range(60):
            if not elapsed_ok():
                break
            if fx == fx and math.isfinite(fx) and abs(fx) <= TOL:
                return x
            h = 1e-6 * max(1.0, abs(x))
            try:
                d = (f(x + h) - fx) / h
            except Exception:
                break
            if d == 0 or d != d or not math.isfinite(d):
                break
            xn = x - fx / d
            if not math.isfinite(xn):
                break
            try:
                fn = f(xn)
            except Exception:
                break
            track(xn, fn)
            if fn == fn and math.isfinite(fn) and abs(fn) <= TOL:
                return xn
            # damp if it overshoots wildly
            if abs(xn - x) > 1e6 * max(1.0, abs(x)):
                xn = 0.5 * (x + xn)
            x, fx = xn, fn
    except Exception:
        pass

    # Return best finite candidate seen (guaranteed finite)
    if best_x == best_x and math.isfinite(best_x) and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
