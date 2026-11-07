"""Root finder for 3 fixed 2D systems using Broyden's method with fallback to SciPy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with line search, falling back to SciPy's root finder.
    """
    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)
        
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def time_up():
        return time.time() > deadline

    # --- Phase 1: Broyden's Method ---
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Perturb if initial point is bad
        x = x + np.array([1e-3, -1e-3])
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            # If still bad, try other perturbations
            for dx in [1e-2, -1e-2, 1e-1, -1e-1]:
                x_try = x0 + np.array([dx, 0])
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)):
                    x = x_try
                    Fx = Fx_try
                    break
            else:
                return x # Return whatever we have, likely fail

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian using forward differences
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h

    best_x = x
    best_norm = norm_Fx

    for _ in range(30):
        if time_up():
            break

        # Solve for step direction
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        # Line search
        alpha = 1.0
        step_accepted = False
        for _ in range(10):
            if time_up():
                break
            
            x_new = x + alpha * d
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
                best_x = x_new

            # Armijo condition: norm_Fx_new < (1 - c * alpha) * norm_Fx
            # c = 1e-4 is standard
            if norm_Fx_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                # Update Broyden
                # J = J + ((Fx_new - Fx - J @ d) @ d.T) / (d @ d)
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-12:
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

    if norm_Fx <= 1e-8:
        return x

    # --- Phase 2: Fallback to SciPy ---
    # Use best_x from Broyden as starting point
    try:
        res = root(F, best_x, method='hybr', options={'maxfev': 1000})
        if res.success:
            return res.x
        # If not successful, check if it's good enough
        if np.linalg.norm(res.fun) <= 1e-8:
            return res.x
        # Return the best of the two
        if np.linalg.norm(res.fun) < best_norm:
            return res.x
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
