"""Root finder for 3 fixed 2-D systems: Broyden's method with global safeguard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's quasi-Newton method.
    
    Strategy:
    1. Try Broyden's method starting from x0 (fast, ~1 F-call/iteration after setup).
    2. If that fails or converges slowly, try a few other starting points via
       a coarse local search, then refine with Broyden's.
    
    Budget: < 500 F-calls, < 120s wall-clock.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    def time_ok():
        return (time.time() - start_time) < time_limit

    def safe_eval_F(x):
        try:
            fx = F(np.asarray(x, dtype=float))
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except Exception:
            return None

    def broyden(x0_init, max_iter=200):
        """Broyden's method with rank-1 updates. Returns (x, n_calls, converged)."""
        n_calls = 0
        x = np.array(x0_init, dtype=float).copy()
        
        # First evaluation
        Fx = safe_eval_F(x)
        n_calls += 1
        if Fx is None:
            return x, n_calls, False
        if np.linalg.norm(Fx) <= 1e-10:
            return x, n_calls, True
        
        # Finite-difference Jacobian for initialization (4 more calls)
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            n_calls += 1
            if Fx_plus is None:
                return x, n_calls, False
            J[:, j] = (Fx_plus - Fx) / h
        
        # Broyden iterations
        for _ in range(max_iter):
            if not time_ok():
                break
            if n_calls >= 490:
                break
            
            # Newton step
            try:
                s = np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                break
            
            # Damping: if step is too large, shrink it
            step_norm = np.linalg.norm(s)
            if step_norm > 1.0:
                s = s * (1.0 / step_norm)
            
            x_new = x - s
            Fx_new = safe_eval_F(x_new)
            n_calls += 1
            if Fx_new is None:
                break
            
            if np.linalg.norm(Fx_new) <= 1e-10:
                return x_new, n_calls, True
            
            # Broyden rank-1 update
            delta_x = x_new - x
            delta_F = Fx_new - Fx
            # Check if update is meaningful
            if np.linalg.norm(delta_x) > 1e-14:
                J = J + (delta_F - J @ delta_x) @ delta_x.T / (delta_x @ delta_x)
            
            x = x_new
            Fx = Fx_new
        
        return x, n_calls, np.linalg.norm(Fx) <= 1e-8

    # Try from x0 first
    x, n_calls, converged = broyden(x0)
    if converged and n_calls < 500:
        return x
    
    # If not converged, try a few nearby points
    best_x = x
    best_norm = np.linalg.norm(safe_eval_F(x) if safe_eval_F(x) is not None else np.array([1e10, 1e10]))
    
    # Generate candidate starting points: perturbations of x0
    candidates = []
    base = np.array(x0, dtype=float)
    
    # Small perturbations
    for dx in [-0.1, 0.1, -1.0, 1.0, -5.0, 5.0]:
        for dy in [-0.1, 0.1, -1.0, 1.0, -5.0, 5.0]:
            candidates.append(base + np.array([dx, dy]))
    
    # Also try the current best x with small perturbations
    for dx in [-0.5, 0.5]:
        for dy in [-0.5, 0.5]:
            candidates.append(best_x + np.array([dx, dy]))
    
    for cand in candidates:
        if not time_ok() or n_calls >= 490:
            break
        
        # Check if this candidate is close to one we already tried
        if any(np.linalg.norm(cand - c) < 1e-6 for c in candidates[:candidates.index(cand)]):
            continue
        
        x_try, calls, conv = broyden(cand, max_iter=100)
        n_calls += calls
        
        if conv:
            return x_try
        
        # Track best
        Fx_try = safe_eval_F(x_try)
        if Fx_try is not None:
            nrm = np.linalg.norm(Fx_try)
            n_calls += 1
            if nrm < best_norm:
                best_norm = nrm
                best_x = x_try
    
    # Final check on best_x
    if best_norm <= 1e-8:
        return best_x
    
    # If we haven't found a root, return the best we have
    return best_x
# EVOLVE-BLOCK-END
