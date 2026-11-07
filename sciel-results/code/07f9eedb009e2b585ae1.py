"""Root finder for 3 fixed 2D systems using Broyden's method with multiple fallbacks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root as scipy_root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Strategy:
      1. Broyden's method with efficient line search from x0
      2. If that fails, try scipy hybr from multiple starting points
      3. Homotopy continuation as last resort
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline

    # --- Phase 1: Broyden's Method from x0 with forward-diff Jacobian ---
    result = _broyden(F, x, start_time, deadline)
    if result is not None:
        return result

    # --- Phase 2: SciPy hybr from multiple starting points ---
    if not time_up():
        starts = [
            x.copy(),
            x + np.array([1.0, 0.0]),
            x + np.array([0.0, 1.0]),
            x + np.array([1.0, 1.0]),
            x + np.array([-1.0, 0.0]),
            x + np.array([0.0, -1.0]),
            x + np.array([0.5, 0.5]),
            x + np.array([-0.5, -0.5]),
        ]
        best_x = x.copy()
        best_norm = np.inf
        for s in starts:
            if time_up():
                break
            try:
                Fx_s = F(s)
                if np.all(np.isfinite(Fx_s)):
                    n = np.linalg.norm(Fx_s)
                    if n < best_norm:
                        best_norm = n
                        best_x = s.copy()
            except Exception:
                pass
        
        for s in starts:
            if time_up():
                break
            try:
                res = scipy_root(F, s, method='hybr', options={'maxfev': 80, 'factor': 100.0})
                if res.success and np.all(np.isfinite(res.x)):
                    Fx_res = F(res.x)
                    if np.all(np.isfinite(Fx_res)):
                        n = np.linalg.norm(Fx_res)
                        if n <= 1e-8:
                            return res.x
                        if n < best_norm:
                            best_norm = n
                            best_x = res.x.copy()
            except Exception:
                pass
        
        if best_norm <= 1e-8:
            return best_x
        if best_norm < np.inf:
            # Polish with Broyden from best_x
            result = _broyden(F, best_x, start_time, deadline)
            if result is not None:
                return result
            return best_x

    # --- Phase 3: Homotopy continuation fallback ---
    if not time_up():
        result = _homotopy_fallback(F, x, start_time, deadline)
        if result is not None:
            return result

    # Last resort: return x0 or best found
    return x


def _broyden(F, x0, start_time, deadline):
    """
    Broyden's method with forward-difference initial Jacobian and Armijo line search.
    Returns x if ||F(x)|| <= 1e-8, else None.
    """
    def time_up():
        return time.time() > deadline

    x = np.array(x0, dtype=float).copy()
    
    # Initial evaluation
    try:
        Fx = F(x)
    except Exception:
        return None
    if not np.all(np.isfinite(Fx)):
        # Try small perturbations
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1]:
            x_try = x + np.array([dx, 0.0])
            try:
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)):
                    x = x_try
                    Fx = Fx_try
                    break
            except Exception:
                continue
        else:
            return None

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian using forward differences (4 calls instead of 6)
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_plus = F(x + e)
            if np.all(np.isfinite(Fx_plus)):
                J[:, j] = (Fx_plus - Fx) / h
            else:
                # Try central if forward fails
                Fx_minus = F(x - e)
                if np.all(np.isfinite(Fx_minus)):
                    J[:, j] = (Fx_plus - Fx_minus) / (2 * h) if np.all(np.isfinite(Fx_plus)) else 0.0
                else:
                    J[:, j] = 0.0
        except Exception:
            J[:, j] = 0.0

    best_x = x.copy()
    best_norm = norm_Fx

    for iteration in range(50):
        if time_up():
            break

        # Check if J is singular
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(d)):
            break

        if np.linalg.norm(d) < 1e-14:
            break

        # Line search with backtracking (Armijo)
        alpha = 1.0
        step_accepted = False
        
        for ls_iter in range(12):
            if time_up():
                break
            
            x_new = x + alpha * d
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            
            try:
                Fx_new = F(x_new)
            except Exception:
                alpha *= 0.5
                continue
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new <= 1e-8:
                return x_new

            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

            # Armijo condition
            if norm_Fx_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                # Broyden update (good variant)
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

    if norm_Fx <= 1e-8:
        return x
    if best_norm <= 1e-8:
        return best_x
    
    # If we got very close, try a few more Newton steps with the current J
    if best_norm < 1e-4 and not time_up():
        x = best_x.copy()
        try:
            Fx = F(x)
            if np.all(np.isfinite(Fx)):
                for _ in range(5):
                    if time_up():
                        break
                    try:
                        d = np.linalg.solve(J, -Fx)
                    except Exception:
                        break
                    if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-14:
                        break
                    x_new = x + d
                    Fx_new = F(x_new)
                    if not np.all(np.isfinite(Fx_new)):
                        break
                    if np.linalg.norm(Fx_new) <= 1e-8:
                        return x_new
                    x = x_new
                    Fx = Fx_new
        except Exception:
            pass
    
    return None


def _homotopy_fallback(F, x_start, start_time, deadline):
    """
    Homotopy/continuation: H_t(x) = (1-t)*(x - anchor) + t*F(x)
    """
    def time_up():
        return time.time() > deadline

    anchors = [
        x_start.copy(),
        x_start + np.array([1.0, 0.0]),
        x_start + np.array([0.0, 1.0]),
        x_start + np.array([1.0, 1.0]),
        x_start - np.array([1.0, 0.0]),
    ]
    
    best_x = x_start.copy()
    best_norm = np.inf
    try:
        Fx_best = F(best_x)
        if np.all(np.isfinite(Fx_best)):
            best_norm = np.linalg.norm(Fx_best)
    except Exception:
        pass

    for anchor in anchors:
        if time_up():
            break
        
        x_curr = anchor.copy()
        t_curr = 0.0
        dt = 0.25  # Larger initial step
        
        for t_step in range(10):
            if time_up():
                break
            
            t_target = min(t_curr + dt, 1.0)
            
            def H_t(x_val, t=t_target, a=anchor):
                return (1 - t) * (x_val - a) + t * F(x_val)
            
            try:
                Hx = H_t(x_curr)
            except Exception:
                dt = max(dt * 0.5, 0.05)
                continue
            
            if not np.all(np.isfinite(Hx)):
                dt = max(dt * 0.5, 0.05)
                continue
            
            norm_Hx = np.linalg.norm(Hx)
            
            # Initialize Jacobian for H_t
            h = 1e-5
            J_h = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Hp = H_t(x_curr + e)
                    if np.all(np.isfinite(Hp)):
                        J_h[:, j] = (Hp - Hx) / h
                    else:
                        J_h[:, j] = 0.0
                except Exception:
                    J_h[:, j] = 0.0
            
            x_broyden = x_curr.copy()
            Fx_broyden = Hx.copy()
            norm_broyden = norm_Hx
            converged = False
            
            for b_iter in range(15):
                if time_up():
                    break
                
                try:
                    d = np.linalg.solve(J_h, -Fx_broyden)
                except np.linalg.LinAlgError:
                    try:
                        d = np.linalg.lstsq(J_h, -Fx_broyden, rcond=None)[0]
                    except Exception:
                        break
                
                if not np.all(np.isfinite(d)):
                    break
                if np.linalg.norm(d) < 1e-12:
                    converged = True
                    break
                
                alpha = 1.0
                step_ok = False
                for ls in range(8):
                    if time_up():
                        break
                    x_trial = x_broyden + alpha * d
                    try:
                        H_trial = H_t(x_trial)
                    except Exception:
                        alpha *= 0.5
                        continue
                    
                    if not np.all(np.isfinite(H_trial)):
                        alpha *= 0.5
                        continue
                    
                    norm_trial = np.linalg.norm(H_trial)
                    if norm_trial <= 1e-10:
                        x_broyden = x_trial
                        Fx_broyden = H_trial
                        norm_broyden = norm_trial
                        converged = True
                        step_ok = True
                        break
                    
                    if norm_trial < (1.0 - 1e-4 * alpha) * norm_broyden:
                        d_dot_d = np.dot(d, d)
                        if d_dot_d > 1e-15:
                            delta_F = H_trial - Fx_broyden - J_h @ d
                            J_h = J_h + np.outer(delta_F, d) / d_dot_d
                        
                        x_broyden = x_trial
                        Fx_broyden = H_trial
                        norm_broyden = norm_trial
                        step_ok = True
                        break
                    alpha *= 0.5
                
                if not step_ok:
                    break
            
            if converged and t_target >= 1.0:
                try:
                    Fx_check = F(x_broyden)
                    if np.all(np.isfinite(Fx_check)):
                        norm_check = np.linalg.norm(Fx_check)
                        if norm_check <= 1e-8:
                            return x_broyden
                        if norm_check < best_norm:
                            best_norm = norm_check
                            best_x = x_broyden.copy()
                except Exception:
                    pass
                break
            elif converged:
                x_curr = x_broyden
                t_curr = t_target
            else:
                dt = max(dt * 0.5, 0.05)

    if best_norm < np.inf:
        return best_x
    return None

# EVOLVE-BLOCK-END
