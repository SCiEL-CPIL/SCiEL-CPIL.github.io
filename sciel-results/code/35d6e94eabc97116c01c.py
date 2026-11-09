import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root of f starting near x0.
    Strategy: Hybrid Inverse Quadratic Interpolation (IQI) with Secant fallback and Global Bracketing.
    1. Initialize with local points.
    2. Iterate:
       - If a bracket exists, use Brent's method (fastest robust convergence).
       - Else, try IQI to predict next point (quadratic fit to last 3 points).
       - If IQI fails or step is too large, use Secant method.
       - If the method diverges (step too large or no improvement), perform a global expanding bracket search.
    3. Return the best point found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    x0 = float(x0)
    
    # State variables
    # We keep a history of the last 3 points (x, f(x)) for IQI
    # We also keep the best point found so far
    best_x = x0
    best_fx = f(best_x)
    best_abs_fx = abs(best_fx)
    eval_count = 1
    
    if best_abs_fx <= 1e-8:
        return best_x
        
    # Initial points for Secant/IQI
    # We need at least 2 points to start Secant, 3 for IQI.
    # Let's generate initial points carefully to avoid immediate failure.
    h = 1e-4 * max(1.0, abs(x0))
    
    x1 = x0
    f1 = best_fx
    
    x2 = x0 + h
    f2 = f(x2)
    eval_count += 1
    if math.isnan(f2) or math.isinf(f2):
        # Try other side
        x2 = x0 - h
        f2 = f(x2)
        eval_count += 1
        if math.isnan(f2) or math.isinf(f2):
            # If we can't even get a second point, return best
            return best_x
    if abs(f2) < best_abs_fx:
        best_x = x2
        best_fx = f2
        best_abs_fx = abs(f2)
        
    # If we have a sign change, we can immediately use Brent
    if f1 * f2 < 0:
        try:
            root = brentq(f, x1, x2, maxiter=50, rtol=1e-12, xtol=1e-12)
            fr = f(root)
            if abs(fr) < best_abs_fx:
                return root
            if abs(fr) <= 1e-8:
                return root
        except Exception:
            pass
            
    # Main loop
    # History: [(x, f(x)), ...]
    history = [(x1, f1), (x2, f2)]
    
    max_iter = 500 - eval_count - 10  # Leave buffer for final checks
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check if we have a bracket in history
        # We can check the last two points, or scan history for a sign change
        # Scanning history is O(N) but N is small. However, checking last two is O(1).
        # Let's check the last two points first.
        if len(history) >= 2:
            x_a, f_a = history[-2]
            x_b, f_b = history[-1]
            if not (math.isnan(f_a) or math.isinf(f_a) or math.isnan(f_b) or math.isinf(f_b)):
                if f_a * f_b < 0:
                    try:
                        root = brentq(f, x_a, x_b, maxiter=50, rtol=1e-12, xtol=1e-12)
                        fr = f(root)
                        if abs(fr) < best_abs_fx:
                            return root
                        if abs(fr) <= 1e-8:
                            return root
                    except Exception:
                        pass
        
        # Try IQI
        x_next = None
        if len(history) >= 3:
            x0_h, f0_h = history[-3]
            x1_h, f1_h = history[-2]
            x2_h, f2_h = history[-1]
            
            # Check for validity
            if not any(math.isnan(fx) or math.isinf(fx) for _, fx in history[-3:]):
                try:
                    # Inverse Quadratic Interpolation
                    # We fit a parabola y(x) through (x0, f0), (x1, f1), (x2, f2)
                    # and find the root of the quadratic.
                    # Formula for the root closest to x2:
                    # Let d1 = f1/f2, d2 = f0/f2
                    # x_next = x2 + 0.5 * f2 * [ (x1-x2)/(f1-f2) + (x0-x1)/(f0-f1) - (x0-x2)/(f0-f2) ]
                    # Wait, that's not the standard formula.
                    # Standard IQI formula:
                    # x_{n+1} = x_n + \frac{f_n}{2} \left( \frac{x_{n-1}-x_n}{f_{n-1}-f_n} + \frac{x_{n-2}-x_{n-1}}{f_{n-2}-f_{n-1}} - \frac{x_{n-2}-x_n}{f_{n-2}-f_n} \right)
                    
                    denom1 = f1_h - f2_h
                    denom2 = f0_h - f1_h
                    denom3 = f0_h - f2_h
                    
                    if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
                        term1 = (x1_h - x2_h) / denom1
                        term2 = (x0_h - x1_h) / denom2
                        term3 = (x0_h - x2_h) / denom3
                        
                        x_next = x2_h + 0.5 * f2_h * (term1 + term2 - term3)
                        
                        # Sanity check: step size
                        if math.isfinite(x_next):
                            step = abs(x_next - x2_h)
                            # Limit step size to avoid flying away
                            max_step = 10.0 * max(1.0, abs(x2_h))
                            if step > max_step:
                                x_next = None # Fallback to Secant
                except Exception:
                    x_next = None
                    
        # Fallback to Secant if IQI failed
        if x_next is None:
            if len(history) >= 2:
                x1_h, f1_h = history[-2]
                x2_h, f2_h = history[-1]
                if not (math.isnan(f1_h) or math.isinf(f1_h) or math.isnan(f2_h) or math.isinf(f2_h)):
                    denom = f2_h - f1_h
                    if abs(denom) > 1e-15:
                        x_next = x2_h - f2_h * (x2_h - x1_h) / denom
                        if math.isfinite(x_next):
                            step = abs(x_next - x2_h)
                            max_step = 10.0 * max(1.0, abs(x2_h))
                            if step > max_step:
                                x_next = None
                    else:
                        x_next = None
            else:
                x_next = None
                
        # If both IQI and Secant failed, or we are stuck, do a global expansion
        if x_next is None:
            # Global expanding bracket from x0
            # We already tried small steps. Let's expand from x0 or current best.
            # Let's try expanding from x0
            d = 1e-2 * max(1.0, abs(x0))
            found_bracket = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                a = x0 - d
                b = x0 + d
                fa = f(a)
                fb = f(b)
                eval_count += 2
                if math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb):
                    d *= 2.0
                    continue
                if abs(fa) < best_abs_fx:
                    best_x = a
                    best_fx = fa
                    best_abs_fx = abs(fa)
                if abs(fb) < best_abs_fx:
                    best_x = b
                    best_fx = fb
                    best_abs_fx = abs(fb)
                if fa * fb < 0:
                    try:
                        root = brentq(f, a, b, maxiter=50, rtol=1e-12, xtol=1e-12)
                        fr = f(root)
                        if abs(fr) < best_abs_fx:
                            return root
                        if abs(fr) <= 1e-8:
                            return root
                    except Exception:
                        pass
                    found_bracket = True
                    break
                d *= 2.0
                if d > 1e100:
                    break
            
            if not found_bracket:
                # If we can't find a bracket, just return best
                return best_x
        
        # Evaluate f(x_next)
        f_next = f(x_next)
        eval_count += 1
        
        if math.isnan(f_next) or math.isinf(f_next):
            # If we got a bad value, let's try a small random perturbation or just stop
            # For now, let's just break and return best
            break
            
        if abs(f_next) < best_abs_fx:
            best_x = x_next
            best_fx = f_next
            best_abs_fx = abs(f_next)
            
        if best_abs_fx <= 1e-8:
            return best_x
            
        # Update history
        history.append((x_next, f_next))
        if len(history) > 3:
            history.pop(0)
            
        # If we are not making progress, we might be stuck in a loop
        # Check if x_next is very close to previous x
        if len(history) >= 2:
            if abs(x_next - history[-2][0]) < 1e-15 * max(1.0, abs(x_next)):
                # Stuck, break
                break

    return best_x
