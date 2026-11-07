"""Root finder for 3 fixed 2D systems using Broyden's Method with Backtracking Line Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    # Candidate starting points: 
    # 1. Provided x0
    # 2. Origin
    # 3. A few diverse points to handle potential local minima/traps, 
    #    but Broyden is fast so we can try a few if the first fails quickly.
    
    starts = [
        x0_arr,
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
        np.array([0.5, 0.5]),
        np.array([-0.5, -0.5]),
    ]
    
    # We will try starts in order. If one converges, we stop.
    # If one fails (diverges or hits max iter without converging), we try the next.
    # Broyden is cheap, so we can afford ~10-20 iterations per start comfortably.
    
    best_x = x0_arr
    best_norm = float('inf')
    total_calls = 0
    max_total_calls = 490  # Reserve 10 calls for safety/final check
    
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
        if total_calls >= max_total_calls:
            break
            
        x_res, norm_res, calls_used = _broyden_solve(F, x_start, max_iter=50, tol=1e-9, max_calls_remaining=max_total_calls - total_calls)
        
        total_calls += calls_used
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
    return best_x

def _broyden_solve(F, x0, max_iter=50, tol=1e-9, max_calls_remaining=490):
    """
    Broyden's method with backtracking line search.
    Returns (x, norm_F, calls_used).
    """
    calls = 0
    x = np.array(x0, dtype=float)
    
    # Initial Evaluation
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return x, float('inf'), 1
        calls += 1
    except Exception:
        return x, float('inf'), 1
        
    norm_F = np.linalg.norm(Fx)
    if norm_F <= tol:
        return x, norm_F, calls
        
    # Initial Jacobian Estimate: Identity (or Finite Diff if we want robustness, but Identity is cheaper)
    # Using Identity is standard for Broyden start if scale is unknown, but FD is safer for initial step direction.
    # Let's use a quick central difference for the FIRST Jacobian only to get a good direction.
    # Cost: 4 calls.
    # If we have budget, do FD. If not, Identity.
    # 4 calls is cheap relative to 490.
    h = 1e-5
    J = np.eye(2)
    try:
        if max_calls_remaining > 10:
            e0 = np.array([h, 0.0])
            e1 = np.array([0.0, h])
            
            Fp0 = np.asarray(F(x + e0), dtype=float).reshape(2)
            Fm0 = np.asarray(F(x - e0), dtype=float).reshape(2)
            Fp1 = np.asarray(F(x + e1), dtype=float).reshape(2)
            Fm1 = np.asarray(F(x - e1), dtype=float).reshape(2)
            calls += 4
            
            if np.all(np.isfinite(Fp0)) and np.all(np.isfinite(Fm0)) and np.all(np.isfinite(Fp1)) and np.all(np.isfinite(Fm1)):
                J = np.zeros((2, 2))
                J[:, 0] = (Fp0 - Fm0) / (2 * h)
                J[:, 1] = (Fp1 - Fm1) / (2 * h)
                if not np.all(np.isfinite(J)) or np.linalg.cond(J) > 1e10:
                    J = np.eye(2)
        else:
            J = np.eye(2)
    except Exception:
        J = np.eye(2)

    # Broyden Iteration
    for _ in range(max_iter):
        if calls >= max_calls_remaining:
            break
        if time.time() - start_time > 145.0:
            break
            
        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, perturb J or use lstsq
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
                
        if not np.all(np.isfinite(s)):
            break
            
        # Backtracking Line Search
        # Start with full step alpha = 1
        alpha = 1.0
        x_trial = x + alpha * s
        F_trial = None
        norm_trial = float('inf')
        
        # Try up to 10 backtracks
        for _ in range(10):
            if calls >= max_calls_remaining:
                break
                
            try:
                F_trial = F(x_trial)
                F_trial = np.asarray(F_trial, dtype=float)
                if F_trial.shape != (2,):
                    F_trial = F_trial.reshape(2)
                if not np.all(np.isfinite(F_trial)):
                    alpha *= 0.5
                    x_trial = x + alpha * s
                    calls += 1
                    continue
                calls += 1
            except Exception:
                alpha *= 0.5
                x_trial = x + alpha * s
                continue
                
            norm_trial = np.linalg.norm(F_trial)
            
            # Armijo-like condition: reduce norm significantly
            # Simple condition: norm_trial < norm_F
            if norm_trial < norm_F:
                break
            else:
                alpha *= 0.5
                x_trial = x + alpha * s
                
        if F_trial is None or not np.all(np.isfinite(F_trial)):
            # If line search failed, try a very small random step? Or just break.
            # Breaking is safer to avoid wasting calls.
            break
            
        # Check convergence
        if norm_trial <= tol:
            return x_trial, norm_trial, calls
            
        # Update Broyden Formula
        # J_new = J + (F_new - F_old - J*s) * s^T / (s^T * s)
        delta_F = F_trial - Fx
        delta_s = x_trial - x # which is alpha * s
        
        # s^T * s
        denom = np.dot(delta_s, delta_s)
        if denom > 1e-12:
            update = (delta_F - J.dot(delta_s)) / denom
            J = J + np.outer(update, delta_s)
            
        x = x_trial
        Fx = F_trial
        norm_F = norm_trial
        
    return x, norm_F, calls
# EVOLVE-BLOCK-END
