"""Root finder for 3 fixed 2D systems using Broyden's method with fallback to SciPy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root as scipy_root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search, falling back to SciPy's root finder.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    # --- Phase 1: Broyden's Method ---
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Perturb if initial point is bad
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]:
            x_try = x + np.array([dx, -dx])
            Fx_try = F(x_try)
            if np.all(np.isfinite(Fx_try)):
                x = x_try
                Fx = Fx_try
                break
        else:
            return x  # Return whatever we have, likely fail

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian using central differences for better accuracy
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = F(x + e)
        Fx_minus = F(x - e)
        if not (np.all(np.isfinite(Fx_plus)) and np.all(np.isfinite(Fx_minus))):
            # Fallback to forward difference
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                J[:, j] = 0.0
            else:
                J[:, j] = (Fx_plus - Fx) / h
        else:
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)

    best_x = x.copy()
    best_norm = norm_Fx

    # Convergence tracking
    prev_norm = norm_Fx
    fast_converging = False

    for iteration in range(40):
        if time_up():
            break

        # Check for fast convergence
        if iteration > 0 and norm_Fx < 0.5 * prev_norm:
            fast_converging = True
        prev_norm = norm_Fx

        # Solve for step direction
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Try pseudo-inverse
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(d)):
            break

        # If step is too small, we're likely at the root or stuck
        if np.linalg.norm(d) < 1e-12:
            break

        # Line search with backtracking
        alpha = 1.0
        step_accepted = False
        
        for ls_iter in range(15):
            if time_up():
                break
            
            x_new = x + alpha * d
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new <= 1e-8:
                return x_new

            # Track best
            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

            # Armijo condition
            if norm_Fx_new < (1.0 - 1e-3 * alpha) * norm_Fx:
                # Update Broyden (good Broyden update)
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-15:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                step_accepted = True
                break
            else:
                alpha *= 0.5

        if not step_accepted:
            break

        # Early termination if converging fast and close
        if fast_converging and norm_Fx < 1e-4:
            if iteration >= 3:
                continue
            break

    if norm_Fx <= 1e-8:
        return x

    # --- Phase 2: Fallback to SciPy ---
    # Use best_x from Broyden as starting point
    if not time_up():
        try:
            res = scipy_root(F, best_x, method='hybrid', options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.success:
                if np.all(np.isfinite(res.x)):
                    return res.x
            # Check if close enough even if not fully converged
            if np.all(np.isfinite(res.x)):
                Fx_res = F(res.x)
                if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) <= 1e-8:
                    return res.x
                # Return best of the two
                if np.linalg.norm(Fx_res) < best_norm:
                    best_x = res.x
                    best_norm = np.linalg.norm(Fx_res)
        except Exception:
            pass

    # --- Phase 3: Try SciPy from original x0 if not tried yet ---
    if not time_up() and np.linalg.norm(best_x - x0) > 1e-10:
        try:
            res = scipy_root(F, x, method='hybrid', options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.success:
                if np.all(np.isfinite(res.x)):
                    return res.x
            if np.all(np.isfinite(res.x)):
                Fx_res = F(res.x)
                if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) <= 1e-8:
                    return res.x
                if np.linalg.norm(Fx_res) < best_norm:
                    best_x = res.x
                    best_norm = np.linalg.norm(Fx_res)
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
