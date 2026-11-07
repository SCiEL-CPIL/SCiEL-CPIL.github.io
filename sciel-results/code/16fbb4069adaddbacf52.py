"""Global optimization + Newton polish root finder for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _norm_F(x, F):
    """Compute ||F(x)||, return inf on error."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return np.inf
        return float(np.linalg.norm(Fx))
    except Exception:
        return np.inf

def _squared_norm_F(x, F):
    """Compute ||F(x)||^2 for optimization."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return 1e100
        return float(np.dot(Fx, Fx))
    except Exception:
        return 1e100

def _newton_polish(F, x0, max_iter=50):
    """
    High-efficiency Newton-Raphson polish.
    Uses forward differences for Jacobian.
    Stops immediately when ||F(x)|| <= 1e-8.
    """
    x = np.array(x0, dtype=float)
    h = 1e-6  # Step size for FD Jacobian
    
    for i in range(max_iter):
        if _time_left() < 0.5:
            return x
        
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return x
            
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x
            
            # Compute Jacobian using forward differences
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = F(x + e)
                Fx_p = np.asarray(Fx_p, dtype=float)
                if Fx_p.shape != (2,) or not np.all(np.isfinite(Fx_p)):
                    return x
                J[:, j] = (Fx_p - Fx) / h
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: gradient descent step
                dx = -0.1 * Fx
                
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                return x
            x = x_new
            
        except Exception:
            return x
            
    return x

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])
    
    best_x = x0.copy()
    best_norm = _norm_F(best_x, F)
    
    # Strategy 1: Try Newton polish directly from x0 first (cheap if x0 is close)
    x_candidate = _newton_polish(F, x0)
    norm_candidate = _norm_F(x_candidate, F)
    if norm_candidate < best_norm:
        best_x = x_candidate
        best_norm = norm_candidate
    
    if best_norm <= 1e-8:
        return best_x
        
    # Strategy 2: Global Optimization using Differential Evolution
    # We search in a box around x0. The size of the box is a heuristic.
    # Since we don't know the scale, we try a few different bounds.
    # DE is stochastic and derivative-free, so it's robust to bad initializations.
    
    # Define bounds for DE. We try a moderate box first.
    # If the root is far, we might need to expand, but DE is good at exploring.
    # We use a fixed seed for reproducibility within the run, but different 
    # strategies might benefit from different seeds.
    
    bounds_list = [
        (x0 - 10, x0 + 10),
        (x0 - 50, x0 + 50),
        (x0 - 100, x0 + 100),
    ]
    
    for bounds in bounds_list:
        if _time_left() < 5.0:
            break
            
        try:
            # Differential Evolution
            # popsize=15 is default, we can tweak for speed/accuracy tradeoff
            # maxiter=100 is default, we limit it to save time
            # tol=1e-10 is strict, but we only need 1e-8 on F, so 1e-10 on F^2 is fine
            # seed=42 for reproducibility
            
            res = differential_evolution(
                _squared_norm_F, 
                bounds, 
                args=(F,),
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-12,
                init='sobol'
            )
            
            if res.success or res.fun < 1e-4: # If DE finds something reasonably good
                x_candidate = res.x
                norm_candidate = _norm_F(x_candidate, F)
                if norm_candidate < best_norm:
                    best_x = x_candidate
                    best_norm = norm_candidate
                    
                if best_norm <= 1e-8:
                    return best_x
                    
                # Polish the result from DE with Newton
                x_polished = _newton_polish(F, best_x)
                norm_polished = _norm_F(x_polished, F)
                if norm_polished < best_norm:
                    best_x = x_polished
                    best_norm = norm_polished
                    
                if best_norm <= 1e-8:
                    return best_x
                    
        except Exception:
            pass

    # Strategy 3: If DE failed or was slow, try Nelder-Mead from x0 and some random points
    if _time_left() > 2.0:
        # Try Nelder-Mead from x0
        try:
            res = minimize(_squared_norm_F, x0, args=(F,), method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            x_candidate = res.x
            norm_candidate = _norm_F(x_candidate, F)
            if norm_candidate < best_norm:
                best_x = x_candidate
                best_norm = norm_candidate
            
            if best_norm <= 1e-8:
                return best_x
                
            x_polished = _newton_polish(F, best_x)
            norm_polished = _norm_F(x_polished, F)
            if norm_polished < best_norm:
                best_x = x_polished
                best_norm = norm_polished
                
            if best_norm <= 1e-8:
                return best_x
        except Exception:
            pass

    # Strategy 4: Random restarts with Newton polish
    if _time_left() > 1.0:
        rng = np.random.default_rng(123)
        for i in range(10):
            if _time_left() < 0.5:
                break
            offset = rng.uniform(-20.0, 20.0, size=2)
            x_init = x0 + offset
            x_candidate = _newton_polish(F, x_init, max_iter=30)
            norm_candidate = _norm_F(x_candidate, F)
            if norm_candidate < best_norm:
                best_x = x_candidate
                best_norm = norm_candidate
            if best_norm <= 1e-8:
                return best_x

    return best_x

# EVOLVE-BLOCK-END
