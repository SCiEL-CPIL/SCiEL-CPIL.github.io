import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x = float(x0)
    
    # Initial points for Secant
    # Choose a small initial step
    h = 1e-4 * max(1.0, abs(x))
    x0_curr = x
    x1_curr = x + h
    
    f0 = None
    f1 = None
    
    try:
        f0 = f(x0_curr)
        if not np.isfinite(f0):
            raise ValueError
        if abs(f0) <= 1e-8:
            return x0_curr
            
        f1 = f(x1_curr)
        if not np.isfinite(f1):
            raise ValueError
        if abs(f1) <= 1e-8:
            return x1_curr
    except Exception:
        # If initial eval fails, fall back to pure bracketing
        pass

    # Secant iterations
    for i in range(20):
        # Check for bracket between current two points
        if f0 is not None and f1 is not None and f0 * f1 < 0:
            try:
                root = brentq(f, x0_curr, x1_curr, xtol=1e-12, rtol=1e-12, maxiter=100)
                fr = f(root)
                if np.isfinite(fr) and abs(fr) <= 1e-8:
                    return root
                else:
                    return root
            except Exception:
                break

        # Calculate next Secant point
        if abs(f1 - f0) < 1e-15:
            break # Derivative approx zero
            
        x2_curr = x1_curr - f1 * (x1_curr - x0_curr) / (f1 - f0)
        
        if not np.isfinite(x2_curr):
            break
            
        # Safeguard: don't go too far
        if abs(x2_curr - x1_curr) > 100 * max(1.0, abs(x1_curr)):
             # If step is huge, maybe we missed the root or diverged.
             # Try to bracket around x1?
             break

        # Shift window
        x0_curr = x1_curr
        f0 = f1
        x1_curr = x2_curr
        
        try:
            f1 = f(x1_curr)
        except Exception:
            break
            
        if not np.isfinite(f1):
            break
            
        if abs(f1) <= 1e-8:
            return x1_curr

    # Fallback: Global Bracketing from original x0
    # This handles cases where Secant diverged or no bracket was found
    def try_brent(center):
        h = 1e-4
        for _ in range(50):
            a = center - h
            b = center + h
            try:
                fa = f(a)
                fb = f(b)
            except:
                return None
            
            if not (np.isfinite(fa) and np.isfinite(fb)):
                h *= 2.0
                continue
                
            if fa == 0:
                return a
            if fb == 0:
                return b
                
            if fa * fb < 0:
                try:
                    root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if abs(f(root)) <= 1e-8:
                        return root
                    else:
                        return root
                except:
                    return None
            
            h *= 2.0
            if h > 1e6:
                break
        return None

    root = try_brent(x)
    if root is not None:
        return root

    root = try_brent(float(x0))
    if root is not None:
        return root

    # Last resort: return best guess
    best_x = x
    best_val = float('inf')
    try:
        if np.isfinite(f0):
            best_val = abs(f0)
            best_x = x0_curr
        if np.isfinite(f1):
            if abs(f1) < best_val:
                best_val = abs(f1)
                best_x = x1_curr
    except:
        pass
        
    return best_x
