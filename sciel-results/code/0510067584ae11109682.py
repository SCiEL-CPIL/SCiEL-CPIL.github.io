"""Root finding for ill-conditioned equations: Inverse Interpolation with History Reuse."""
import math
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin below 120s limit
    MAX_EVALS = 490   # Safety margin below 500 limit
    
    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    # History of (x, f(x)) pairs for inverse interpolation
    history = []

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS:
            return True
        if time.time() - start_time > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, history
        if check_budget():
            return None
        try:
            val = f(x)
            eval_count += 1
            if val is None or not np.isfinite(val):
                return None
            val = float(val)
            if best_f is None or abs(val) < abs(best_f):
                best_f = val
                best_x = x
            # Add to history, keeping unique x values (approx)
            # We want to keep the most recent and most informative points
            if len(history) == 0 or abs(x - history[-1][0]) > 1e-12:
                history.append((x, val))
                # Limit history size to avoid numerical issues in interpolation
                if len(history) > 10:
                    # Keep the 3 most recent, and maybe a few others?
                    # For now, just truncate the oldest
                    history.pop(0)
            return val
        except Exception:
            return None

    # Initial evaluation
    fx0 = safe_eval(best_x)
    if fx0 is None:
        return best_x
    if abs(fx0) <= 1e-8:
        return best_x

    # Strategy 1: Inverse Interpolation (Illinois-like but with quadratic/cubic model)
    # We maintain a bracket [a, b] if possible, or a local region if not.
    # Since we don't have a bracket initially, we try to find one or use local interpolation.
    
    # Let's try to establish a bracket first using a few steps, but efficiently.
    # If we can't bracket, we fall back to minimizing |f(x)| using the interpolation model.
    
    a = best_x
    fa = best_f
    b = best_x
    fb = best_f
    
    # Try to find a sign change quickly
    # We will use the history to guide this if possible, but let's start with simple expansion.
    # However, the directive asks for a structural change: "history-dependent sequential prediction".
    # So, we will primarily use the interpolation model to propose the next x.
    
    # If we have no bracket, we need to generate points.
    # Let's use a strategy that combines:
    # 1. If we have a bracket, use inverse quadratic/cubic interpolation.
    # 2. If we don't, use the model to predict a point that minimizes |f(x)| locally,
    #    or expand the search region.
    
    # To start, let's get 3 points to build a quadratic model.
    # We already have (x0, f(x0)).
    # Let's evaluate x0 + h and x0 - h to get a local picture.
    
    h = 1e-4 * max(1.0, abs(x0))
    
    # Evaluate right
    x_r = x0 + h
    f_r = safe_eval(x_r)
    if f_r is None:
        return best_x
    
    # Evaluate left
    x_l = x0 - h
    f_l = safe_eval(x_l)
    if f_l is None:
        return best_x

    # Now we have 3 points: (x_l, f_l), (x0, f0), (x_r, f_r)
    # Check for sign changes
    if f_l * f0 < 0:
        a, fa = x_l, f_l
        b, fb = x0, f0
    elif f0 * f_r < 0:
        a, fa = x0, f0
        b, fb = x_r, f_r
    elif f_l * f_r < 0:
        # Sign change across x0, but x0 is not a root? Impossible if f is continuous.
        # Maybe x0 is the root?
        if abs(f0) < abs(f_l) and abs(f0) < abs(f_r):
             a, fa = x_l, f_l
             b, fb = x_r, f_r
        else:
             # This case is tricky. Let's just pick the interval with the smaller |f| endpoints?
             # Actually, if f_l and f_r have opposite signs, there is a root in [x_l, x_r].
             a, fa = x_l, f_l
             b, fb = x_r, f_r
    else:
        # No sign change. We are in a local extremum or monotonic region without a root nearby.
        # We will try to minimize |f(x)| using the model.
        # But for now, let's just proceed with the best point found so far as a "degenerate" bracket
        # or switch to a minimization mode.
        # Given the task is "root finding", if no root exists, we should return the point with min |f(x)|.
        # The interpolation model can be used to find a local minimum of |f(x)|.
        
        # Let's try to expand the search region to find a sign change.
        # We will double the step size.
        step = h
        found_bracket = False
        for _ in range(10):
            if check_budget():
                break
            x_try_r = x0 + step
            f_try_r = safe_eval(x_try_r)
            if f_try_r is not None:
                if f0 * f_try_r < 0:
                    a, fa = x0, f0
                    b, fb = x_try_r, f_try_r
                    found_bracket = True
                    break
                if abs(f_try_r) <= 1e-8:
                    return x_try_r
            
            x_try_l = x0 - step
            f_try_l = safe_eval(x_try_l)
            if f_try_l is not None:
                if f0 * f_try_l < 0:
                    a, fa = x_try_l, f_try_l
                    b, fb = x0, f0
                    found_bracket = True
                    break
                if abs(f_try_l) <= 1e-8:
                    return x_try_l
            
            step *= 2.0
            if step > 1e6:
                break
        
        if not found_bracket:
            # No bracket found. Return best_x.
            # We could try Nelder-Mead here, but let's stick to the interpolation strategy first.
            return best_x

    # We have a bracket [a, b]
    # Now use inverse interpolation to converge.
    # We maintain a list of points in the bracket.
    # The history list currently contains points outside the bracket potentially.
    # Let's filter history to only include points within [a, b] or close to it.
    # Actually, it's better to restart the history with the bracket endpoints and any internal points.
    
    # Let's clear history and add the bracket points
    history = []
    if a != best_x:
        history.append((a, fa))
    history.append((best_x, best_f)) # Might be inside or outside
    if b != best_x:
        history.append((b, fb))
    
    # Ensure history is sorted by x
    history.sort(key=lambda p: p[0])
    
    # Iterate
    for _ in range(100):
        if check_budget():
            break
        
        # Get the 3 most recent points in the bracket for quadratic interpolation
        # Or use all points if we want higher order?
        # Let's use the 3 points closest to the current best estimate.
        # For simplicity, let's use the 3 most recent points added to the bracket.
        
        # We need to select points. Let's use the points that define the current sub-interval.
        # Brent's method logic:
        # We have a, b, c (previous b), d (previous c), e (previous step).
        # But we want to use inverse interpolation.
        
        # Let's use a simpler approach:
        # Use the 3 points in the bracket that are most recent.
        # If we have fewer than 3, use secant.
        
        # Let's implement a robust inverse quadratic interpolation step.
        # Points: x1, x2, x3 with f(x1), f(x2), f(x3)
        # We want to find x such that f(x) = 0.
        # Using Lagrange interpolation on 1/f or directly on f?
        # Directly on f is standard.
        
        # Let's pick the 3 points from history that are within [a, b] and most recent.
        # Since history is small, we can just use the last 3.
        
        if len(history) < 2:
            # Should not happen if we have a bracket
            break
            
        if len(history) == 2:
            # Secant method
            x1, f1 = history[-2]
            x2, f2 = history[-1]
            if f1 == f2:
                break
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            # Inverse Quadratic Interpolation
            x1, f1 = history[-3]
            x2, f2 = history[-2]
            x3, f3 = history[-1]
            
            # Check for duplicate x values
            if abs(x1 - x2) < 1e-12 or abs(x2 - x3) < 1e-12 or abs(x1 - x3) < 1e-12:
                # Fall back to secant
                x1, f1 = history[-2]
                x2, f2 = history[-1]
                if f1 == f2:
                    break
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            else:
                # Lagrange basis for f(x) = 0
                # L1(x) = (x-x2)(x-x3)/((x1-x2)(x1-x3))
                # L2(x) = (x-x1)(x-x3)/((x2-x1)(x2-x3))
                # L3(x) = (x-x1)(x-x2)/((x3-x1)(x3-x2))
                # f(x) = f1*L1 + f2*L2 + f3*L3 = 0
                # This is a quadratic in x.
                
                # Coefficients of x^2, x, 1
                # f1*(x^2 - (x2+x3)x + x2*x3) / D1
                # f2*(x^2 - (x1+x3)x + x1*x3) / D2
                # f3*(x^2 - (x1+x2)x + x1*x2) / D3
                
                D1 = (x1 - x2) * (x1 - x3)
                D2 = (x2 - x1) * (x2 - x3)
                D3 = (x3 - x1) * (x3 - x2)
                
                A = f1/D1 + f2/D2 + f3/D3
                B = -f1*(x2+x3)/D1 - f2*(x1+x3)/D2 - f3*(x1+x2)/D3
                C = f1*x2*x3/D1 + f2*x1*x3/D2 + f3*x1*x2/D3
                
                if abs(A) < 1e-12:
                    # Linear
                    if abs(B) < 1e-12:
                        break
                    x_new = -C / B
                else:
                    # Quadratic formula
                    disc = B*B - 4*A*C
                    if disc < 0:
                        # No real root, fall back to bisection or secant
                        x_new = (a + b) / 2.0
                    else:
                        sqrt_disc = math.sqrt(disc)
                        root1 = (-B + sqrt_disc) / (2*A)
                        root2 = (-B - sqrt_disc) / (2*A)
                        
                        # Choose the root that is in the bracket [a, b]
                        # Or closest to the last point x3?
                        # Let's check which one is in [a, b]
                        in_bracket_1 = a <= root1 <= b
                        in_bracket_2 = a <= root2 <= b
                        
                        if in_bracket_1 and in_bracket_2:
                            # Choose the one closer to x3
                            if abs(root1 - x3) < abs(root2 - x3):
                                x_new = root1
                            else:
                                x_new = root2
                        elif in_bracket_1:
                            x_new = root1
                        elif in_bracket_2:
                            x_new = root2
                        else:
                            # Neither in bracket, use bisection
                            x_new = (a + b) / 2.0
        
        # Ensure x_new is within [a, b]
        if x_new < a or x_new > b:
            x_new = (a + b) / 2.0
            
        # Evaluate f(x_new)
        f_new = safe_eval(x_new)
        if f_new is None:
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update bracket
        if fa * f_new < 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new
            
        # Update history
        history.append((x_new, f_new))
        if len(history) > 10:
            # Keep the points that are in the current bracket [a, b]
            # and the most recent ones
            new_history = []
            for p in history:
                if a <= p[0] <= b:
                    new_history.append(p)
            # If we have too many, keep the last 5
            if len(new_history) > 5:
                new_history = new_history[-5:]
            # If we have fewer than 3, add some from the old history that are close
            # This is getting complex. Let's just keep the last 5 overall.
            history = history[-5:]
            
        # Check for convergence of x
        if abs(b - a) < 1e-12:
            return (a + b) / 2.0

    # If loop finishes, try Brent's method on the final bracket if it exists
    if fa * fb < 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=20)
            return root
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
