import math
import time
import numpy as np
from scipy.optimize import brentq

# Global deadline: evaluator kills after 150s wall-clock.
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490  # stay under 500

def _time_left():
    return _DEADLINE - time.time()

def _counted_call(f, x, counter):
    """Call f(x), track count, guard against exceptions / non-finite."""
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val

def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: 
    1. Check x0 directly.
    2. Estimate local scale/derivative to determine adaptive step size.
    3. Perform an adaptive grid search to find a bracket [a, b] with sign change.
    4. Refine the root using Secant/Newton iterations first, falling back to brentq.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    f0 = _counted_call(f, x0, counter)
    if math.isfinite(f0) and abs(f0) <= 1e-8:
        return x0

    if not math.isfinite(f0):
        # If f(x0) is invalid, we must search elsewhere. 
        # We'll start the adaptive scan from x0 anyway, but skip using f0 for scaling.
        pass

    # 2. Estimate local scale
    # We want to estimate h such that f(x0+h) is significantly different from f(x0).
    # Try a small step. If f0 is finite, use it.
    if math.isfinite(f0):
        # Try a few small steps to estimate derivative magnitude
        # h_est = |f(x0)| / |f'(x0)| approx
        # We don't know f'(x0) directly, but we can probe.
        probe_steps = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        scale_est = 1.0  # default scale
        
        for h in probe_steps:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 20:
                break
            x_probe = x0 + h
            f_probe = _counted_call(f, x_probe, counter)
            if math.isfinite(f_probe):
                df = abs(f_probe - f0)
                if df > 1e-6:  # Significant change
                    # Estimate scale as h * (|f0| / df) roughly? 
                    # Or just use h as a baseline for the local period.
                    # If df is small, the function is flat, so we can use larger steps.
                    # If df is large, the function is steep, so we need smaller steps.
                    # Let's say the "characteristic length" is h * (1 + |f0|/df)
                    # But simpler: if we see a change of df over h, the root is likely within h * (|f0|/df) if linear.
                    # Let's set the base step to be proportional to h, but adjusted.
                    # If |f0| is small, root is close.
                    # If |f0| is large, root is further.
                    # Let's just record that we have a valid probe.
                    # We will use a grid step that is a fraction of this h.
                    # If h=1e-4 and df is large, step should be small.
                    # If h=1.0 and df is small, step can be larger.
                    # Heuristic: step = h * 0.5
                    scale_est = h * 0.5
                    break
        else:
            # No significant change found in probes, function is very flat or we ran out of budget
            # Use a default scale
            scale_est = 1.0
    else:
        # f0 is not finite, we can't estimate scale from it.
        # Assume a default scale.
        scale_est = 1.0

    # 3. Adaptive Grid Search
    # We will scan outwards from x0.
    # We maintain a list of points and their f values.
    # We generate points on a grid with step = scale_est * multiplier.
    # If no bracket is found, we increase the multiplier (coarser grid) and expand the range.
    
    # Initial step size
    base_step = max(scale_est, 1e-6)
    
    # We will try several "rings" of increasing radius and step size.
    # Ring 0: Radius 10*base_step, Step 0.1*base_step
    # Ring 1: Radius 100*base_step, Step 1.0*base_step
    # Ring 2: Radius 1000*base_step, Step 10.0*base_step
    # Ring 3: Radius 10000*base_step, Step 100.0*base_step
    
    # But we must be careful not to miss roots if the function oscillates with a period
    # much smaller than base_step. However, base_step was estimated from the local slope.
    # If the function is highly oscillatory, the local slope estimate might be noisy.
    # To be safe, we start with a very fine step relative to the estimated scale.
    
    rings = [
        (10.0, 0.1),   # radius multiplier, step multiplier
        (100.0, 1.0),
        (1000.0, 10.0),
        (10000.0, 100.0)
    ]
    
    points = []
    if math.isfinite(f0):
        points.append((x0, f0))
    
    bracket_found = False
    bracket_a, bracket_b = None, None
    
    for radius_mult, step_mult in rings:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
            
        radius = radius_mult * base_step
        step = step_mult * base_step
        
        # Ensure step is not too small to cause float issues or too large
        if step < 1e-8:
            step = 1e-8
        if step > 1e6:
            step = 1e6
            
        # Generate points in [x0 - radius, x0 + radius]
        # Number of points: 2 * radius / step
        n_points = int(2 * radius / step) + 1
        # Cap n_points to avoid huge memory/time usage
        if n_points > 500:
            # Adjust step to fit within 500 points
            step = 2 * radius / 499
            n_points = 500
            
        a = x0 - radius
        b = x0 + radius
        xs = np.linspace(a, b, n_points)
        
        prev_x = None
        prev_f = None
        
        # We already have x0. If x0 is in this grid, we might re-evaluate it.
        # To save calls, we can check if x0 is close to any grid point.
        # But for simplicity and robustness against float errors, we just evaluate.
        # However, we can skip evaluation if we already have the value.
        # Let's just evaluate. The cost is low.
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
                
            # Check if we already evaluated this x (or close to it)
            # Simple check: if x is close to x0 and we have f0
            if math.isclose(x, x0, rel_tol=1e-12, abs_tol=1e-12) and math.isfinite(f0):
                fx = f0
            else:
                fx = _counted_call(f, x, counter)
            
            if math.isfinite(fx):
                if abs(fx) <= 1e-8:
                    return x
                
                if prev_f is not None and math.isfinite(prev_f):
                    if prev_f * fx < 0:
                        # Bracket found
                        bracket_found = True
                        bracket_a = prev_x
                        bracket_b = x
                        break
                
                prev_x = x
                prev_f = fx
            else:
                prev_x = None
                prev_f = None
        
        if bracket_found:
            break
    
    if not bracket_found:
        # If no bracket found, we are in trouble.
        # Return x0 as a fallback, or the point with smallest |f(x)| seen.
        # We didn't track min |f(x)| explicitly, but we can just return x0.
        # Or, if we have points, find the one with min |f(x)|.
        # For now, return x0.
        return x0

    # 4. Refine the root
    # We have a bracket [bracket_a, bracket_b].
    # Try Secant Method first, as it might converge faster than brentq for smooth functions.
    # Secant method needs two points. We have f(bracket_a) and f(bracket_b) (from prev_f and fx).
    # Let's retrieve them or re-evaluate if necessary (but we should have them).
    # To be safe, let's just use brentq which is robust.
    # However, the prompt says "rethinking the approach". 
    # Using brentq is standard. 
    # Let's try a few Secant iterations manually to save calls if possible.
    
    a = bracket_a
    b = bracket_b
    fa = None
    fb = None
    
    # We need fa and fb. We don't have them stored in a variable scope outside the loop easily.
    # Let's re-evaluate a and b to be sure. This costs 2 calls.
    if _time_left() > 1.0 and counter[0] < _CALL_BUDGET - 2:
        fa = _counted_call(f, a, counter)
        fb = _counted_call(f, b, counter)
    else:
        # Fallback to brentq which will evaluate them
        try:
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
        return x0

    if not (math.isfinite(fa) and math.isfinite(fb)):
        try:
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
        return x0

    # Secant Method Iterations
    # x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    x1 = a
    x2 = b
    f1 = fa
    f2 = fb
    
    for _ in range(10):
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
            break
            
        if abs(f2 - f1) < 1e-15:
            break
            
        x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
        
        # Check if x3 is within the bracket? 
        # Secant can jump out. If it does, we should switch to brentq or bisection.
        # For safety, if x3 is not between x1 and x2, we stop and use brentq.
        if not (min(x1, x2) <= x3 <= max(x1, x2)):
            break
            
        f3 = _counted_call(f, x3, counter)
        
        if not math.isfinite(f3):
            break
            
        if abs(f3) <= 1e-8:
            return x3
            
        # Update points
        # Keep the two points that bracket the root? 
        # Secant doesn't maintain a bracket. 
        # If we want to maintain a bracket, we should use a method like Illinois or Bisection-Secant hybrid.
        # Since we already have a bracket [a,b], let's just use brentq if Secant doesn't converge in 2-3 steps.
        # Actually, just using brentq is safer and likely faster in terms of code complexity and robustness.
        # The "different strategy" is the Adaptive Grid. The refinement can be standard.
        # Let's just break out and use brentq for the final convergence to ensure we meet the tolerance.
        break
        
    # Fall back to brentq for robust convergence
    try:
        result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        if abs(f(result)) <= 1e-8:
            return result
    except Exception:
        pass

    # If brentq fails, return the point with smallest |f| from our attempts?
    # We don't have that easily accessible. Return x0 or the last secant point?
    # Return x0 as the safest fallback that was checked first.
    return x0
