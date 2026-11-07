"""Root finder for 3 fixed two-dimensional systems using scipy least_squares with multiple methods."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses scipy.optimize.least_squares with multiple strategies.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    x0_arr = np.array(x0, dtype=float)
    
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    def update_best(x, Fx):
        nonlocal best_x, best_norm
        norm = float(np.linalg.norm(Fx))
        if norm < best_norm:
            best_x = x.copy()
            best_norm = norm
        if best_norm <= 1e-8:
            return True
        return False
    
    # Evaluate initial point
    try:
        Fx0 = np.array(F(x0_arr), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            Fx0 = np.array(F(x0_arr), dtype=float)
        update_best(x0_arr, Fx0)
        if best_norm <= 1e-8:
            return best_x
    except Exception:
        pass

    # Try multiple methods and strategies
    methods = ['lm', 'trf', 'dogbox']
    x0_variants = [x0_arr]
    
    # Add some perturbed starting points
    for scale in [0.1, 1.0, 10.0]:
        for dx in [scale, -scale]:
            x0_variants.append(x0_arr + np.array([dx, 0.0]))
            x0_variants.append(x0_arr + np.array([0.0, dx]))
    
    # Limit variants to keep call count reasonable
    x0_variants = x0_variants[:6]
    
    for method in methods:
        if time.time() > deadline:
            break
        for x0v in x0_variants:
            if time.time() > deadline:
                break
            if best_norm <= 1e-8:
                return best_x
            
            try:
                res = least_squares(
                    F,
                    x0v,
                    method=method,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=200,
                )
                x_res = res.x
                Fx_res = np.array(F(x_res), dtype=float)
                converged = update_best(x_res, Fx_res)
                if converged:
                    return best_x
            except Exception:
                continue

    # If still not converged, try Broyden's method as fallback
    if time.time() < deadline - 10 and best_norm > 1e-8:
        x = best_x.copy()
        n = 2
        
        try:
            Fx = np.array(F(x), dtype=float)
            norm_Fx = float(np.linalg.norm(Fx))
            if norm_Fx <= 1e-8:
                return x
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx

            # Initialize Jacobian via finite differences
            h = 1e-5
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = h
                Fp = np.array(F(x + e), dtype=float)
                Fm = np.array(F(x - e), dtype=float)
                J[:, j] = (Fp - Fm) / (2 * h)

            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline:
                    break

                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    try:
                        dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    except:
                        break

                alpha = 1.0
                max_ls = 10
                ls_success = False
                for _ in range(max_ls):
                    x_new = x + alpha * dx
                    Fx_new = np.array(F(x_new), dtype=float)
                    norm_Fx_new = float(np.linalg.norm(Fx_new))

                    if norm_Fx_new <= 1e-8:
                        return x_new

                    if norm_Fx_new < best_norm:
                        best_x = x_new.copy()
                        best_norm = norm_Fx_new

                    if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                        s = x_new - x
                        y = Fx_new - Fx
                        s_dot_s = float(np.dot(s, s))
                        if s_dot_s > 1e-30:
                            J += np.outer(y - J @ s, s) / s_dot_s
                        x = x_new
                        Fx = Fx_new
                        norm_Fx = norm_Fx_new
                        ls_success = True
                        break

                    alpha *= 0.5

                if not ls_success:
                    break

                if norm_Fx <= 1e-8:
                    return x
        except Exception:
            pass

    # Return best result found
    if not np.all(np.isfinite(best_x)):
        return np.array(x0, dtype=float)
    return best_x
# EVOLVE-BLOCK-END
