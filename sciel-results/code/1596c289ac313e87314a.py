import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(val):
    try:
        return math.isfinite(val)
    except:
        return False

def _illinois_solve(f, a, b, fa, fb, tol=1e-10, maxiter=50):
    """Illinois method to find root in bracket [a,b] where f(a)*f(b)<0."""
    if fa * fb > 0:
        return None
    
    for _ in range(maxiter):
        if _elapsed() > 140.0:
            break
        
        # Illinois interpolation
        if abs(fa) < abs(fb):
            # Interpolate closer to a
            x = (a * fb - b * fa) / (fb - fa)
        else:
            # Interpolate closer to b
            x = (a * fb - b * fa) / (fb - fa)
        
        # Ensure x is in [a, b]
        if x <= a or x >= b:
            x = (a + b) / 2.0
        
        try:
            fx = f(x)
        except:
            fx = float('inf')
        
        if not _is_finite(fx):
            # Fall back to bisection
            x = (a + b) / 2.0
            try:
                fx = f(x)
            except:
                fx = float('inf')
            if not _is_finite(fx):
                break
        
        if abs(fx) <= tol:
            return x
        
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
            fa *= 0.5  # Illinois damping
        
        if abs(b - a) < 1e-12:
            break
    
    # Return the better of a, b
    if abs(fa) < abs(fb):
        return a
    else:
        return b

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    best_x = x0
    best_f = None
    
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')

    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return x0

    if best_f is None or not _is_finite(best_f):
        best_f = float('inf')

    # Strategy 1: Global multi-scale grid search to find sign-change brackets
    # Evaluate f at exponentially spaced points across a wide domain
    
    # Define search scales: small to large
    base_steps = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    # Store all evaluated points for bracket detection
    points = []  # list of (x, f(x))
    
    # Add x0 first
    if _is_finite(best_f):
        points.append((x0, best_f))
    
    # Evaluate at exponentially spaced points in both directions
    for direction in [1, -1]:
        if _elapsed() > 135.0:
            break
            
        prev_x = x0
        prev_f = best_f
        
        for s in base_steps:
            if _elapsed() > 135.0:
                break
                
            target_x = x0 + direction * s
            
            try:
                target_f = f(target_x)
            except:
                target_f = float('inf')
            
            if not _is_finite(target_f):
                continue
            
            # Update best
            if abs(target_f) < abs(best_f):
                best_f = target_f
                best_x = target_x
                
            if abs(target_f) <= 1e-8:
                return target_x
            
            # Add to points list
            points.append((target_x, target_f))
            
            # Check for sign change with previous point
            if _is_finite(prev_f) and prev_f * target_f < 0:
                # Bracket found! Use Illinois method
                root = _illinois_solve(f, prev_x, target_x, prev_f, target_f, tol=1e-10, maxiter=50)
                if root is not None and _is_finite(root):
                    try:
                        f_root = f(root)
                        if _is_finite(f_root) and abs(f_root) <= 1e-8:
                            return root
                    except:
                        pass
            
            # Update prev for next step
            prev_x = target_x
            prev_f = target_f

    # Strategy 2: If no bracket found via simple expansion, try a denser grid
    # around the region where |f| was smallest
    if _elapsed() < 130.0:
        # Find the region around best_x to search more densely
        center = best_x
        
        # Try a few different grid densities
        for grid_size in [10, 20, 50]:
            if _elapsed() > 135.0:
                    break
                    
            # Search in a window around center
            window = 1.0
            a = center - window
            b = center + window
            
            # Evaluate on a uniform grid
            xs = np.linspace(a, b, grid_size)
            grid_points = []
            
            for x in xs:
                if _elapsed() > 135.0:
                    break
                    
                try:
                    fx = f(x)
                except:
                    fx = float('inf')
                
                if _is_finite(fx):
                    grid_points.append((x, fx))
                    
                    if abs(fx) < abs(best_f):
                        best_f = fx
                        best_x = x
                        
                    if abs(fx) <= 1e-8:
                        return x
            
            # Check for sign changes in the grid
            for i in range(len(grid_points) - 1):
                x1, f1 = grid_points[i]
                x2, f2 = grid_points[i + 1]
                
                if f1 * f2 < 0:
                    root = _illinois_solve(f, x1, x2, f1, f2, tol=1e-10, maxiter=50)
                    if root is not None and _is_finite(root):
                        try:
                            f_root = f(root)
                            if _is_finite(f_root) and abs(f_root) <= 1e-8:
                                return root
                        except:
                            pass

    # Strategy 3: Local minimization of f(x)^2 if no bracket found
    if _elapsed() < 138.0:
        bounds_list = [
            (x0 - 1.0, x0 + 1.0),
            (x0 - 10.0, x0 + 10.0),
            (x0 - 100.0, x0 + 100.0),
            (x0 - 1000.0, x0 + 1000.0)
        ]

        for a, b in bounds_list:
            if _elapsed() > 142.0:
                break
                
            if a > b:
                a, b = b, a
                
            try:
                def squared_residual(x):
                    try:
                        val = f(x)
                        if not _is_finite(val):
                            return 1e100
                        return val * val
                    except:
                        return 1e100

                res = minimize_scalar(squared_residual, bounds=(a, b), method='bounded', 
                                     options={'maxiter': 50, 'xatol': 1e-12})
                if res.success or _is_finite(res.fun):
                    x_candidate = res.x
                    try:
                        f_candidate = f(x_candidate)
                    except:
                        f_candidate = float('inf')
                    
                    if _is_finite(f_candidate):
                        if abs(f_candidate) < abs(best_f):
                            best_f = f_candidate
                            best_x = x_candidate
                            
                        if abs(f_candidate) <= 1e-8:
                            return x_candidate
            except:
                continue

    return best_x
