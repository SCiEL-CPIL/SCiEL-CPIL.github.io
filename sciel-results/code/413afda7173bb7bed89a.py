"""Root finder for 3 fixed two-dimensional systems using Broyden's Method with line search and restarts."""
# EVOLVE-BLOCK-START
import numpy as np


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _initial_jacobian(F, x, Fx):
    """Compute Jacobian using central differences."""
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp = _safe_F(F, x + e)
        Fm = _safe_F(F, x - e)
        if Fp is not None and Fm is not None:
            J[:, j] = (Fp - Fm) / (2 * h)
        else:
            # Fallback to forward difference
            if Fp is not None:
                J[:, j] = (Fp - Fx) / h
            elif Fm is not None:
                J[:, j] = (Fx - Fm) / h
            else:
                return None
    return J


def _broyden_solve(F, x0, max_iter=100):
    """
    Solve F(x) = 0 using Broyden's Method.
    Returns (x, Fx, success, num_calls)
    """
    x = np.array(x0, dtype=float, copy=True)
    Fx = _safe_F(F, x)
    if Fx is None:
        return x, None, False, 1
    
    num_calls = 1
    if np.linalg.norm(Fx) <= 1e-12:
        return x, Fx, True, num_calls

    J = _initial_jacobian(F, x, Fx)
    if J is None:
        return x, Fx, False, num_calls + 4
    
    num_calls += 4 # 4 calls for central diff (2 points * 2 dims)
    
    for it in range(max_iter):
        if np.linalg.norm(Fx) <= 1e-12:
            return x, Fx, True, num_calls
        
        # Solve J @ s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                return x, Fx, False, num_calls
        
        # Line search
        alpha = 1.0
        x_trial = x + alpha * s
        Fx_trial = _safe_F(F, x_trial)
        num_calls += 1
        
        Fx_norm = np.linalg.norm(Fx)
        
        # Backtracking
        max_ls = 20
        ls_count = 0
        while Fx_trial is None or np.linalg.norm(Fx_trial) > 0.9 * Fx_norm:
            alpha *= 0.5
            x_trial = x + alpha * s
            Fx_trial = _safe_F(F, x_trial)
            num_calls += 1
            ls_count += 1
            if ls_count > max_ls or alpha < 1e-12:
                break
        
        if Fx_trial is None:
            return x, Fx, False, num_calls
            
        # If no reduction, fail this attempt
        if np.linalg.norm(Fx_trial) >= Fx_norm:
            return x, Fx, False, num_calls

        # Update x and Fx
        x_new = x_trial
        Fx_new = Fx_trial
        
        # Broyden Update: J_new = J + (f_new - f_old - J*s) * s^T / (s^T*s)
        # s is the step taken: x_new - x
        step = x_new - x
        denom = np.dot(step, step)
        if denom > 1e-12:
            delta_f = Fx_new - Fx - J @ step
            J = J + np.outer(delta_f, step) / denom
            
        x = x_new
        Fx = Fx_new
        
        if np.linalg.norm(Fx) <= 1e-10:
            return x, Fx, True, num_calls

    return x, Fx, np.linalg.norm(Fx) <= 1e-8, num_calls


def find_root_system(F, x0):
    x = np.array(x0, dtype=float, copy=True)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # First try from x0
    x_best, Fx_best, success, calls = _broyden_solve(F, x)
    best_norm = np.linalg.norm(Fx_best) if Fx_best is not None else np.inf
    total_calls = calls
    
    if success and best_norm <= 1e-8:
        return x_best

    # If failed, try random restarts
    # We limit restarts to keep total calls low
    rng = np.random.default_rng(42)
    
    # Strategy: Try a few points in a local box, then wider
    scales = [1.0, 5.0, 10.0]
    
    for scale in scales:
        for _ in range(5):
            # Generate random point
            x_start = x0 + rng.uniform(-scale, scale, size=2)
            
            x_candidate, Fx_candidate, success, calls = _broyden_solve(F, x_start)
            total_calls += calls
            
            if Fx_candidate is not None:
                norm_cand = np.linalg.norm(Fx_candidate)
                if norm_cand < best_norm:
                    best_norm = norm_cand
                    x_best = x_candidate
                    if success and best_norm <= 1e-8:
                        return x_best
            
            # Early termination if we've used too many calls
            if total_calls > 300:
                break
        if total_calls > 300:
            break
            
    return x_best
# EVOLVE-BLOCK-END
