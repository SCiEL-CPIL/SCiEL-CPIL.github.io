"""Root finder for 3 fixed 2D systems using Broyden's method with homotopy fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root as scipy_root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary: Broyden's method with adaptive line search.
    Fallback: Homotopy/continuation from anchor points, then Broyden refinement.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    # --- Phase 1: Broyden's Method from x0 ---
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
            # Go straight to homotopy fallback
            return _homotopy_fallback(F, x, start_time, deadline)

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian using central differences
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = F(x + e)
        Fx_minus = F(x - e)
        if not (np.all(np.isfinite(Fx_plus)) and np.all(np.isfinite(Fx_minus))):
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                J[:, j] = 0.0
            else:
                J[:, j] = (Fx_plus - Fx) / h
        else:
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)

    best_x = x.copy()
    best_norm = norm_Fx
    prev_norm = norm_Fx

    for iteration in range(50):
        if time_up():
            break

        # Solve for step direction
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(d)):
            break

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

            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

            # Armijo condition
            if norm_Fx_new < (1.0 - 1e-3 * alpha) * norm_Fx:
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

        # Early termination if very close
        if norm_Fx < 1e-4 and iteration >= 5:
            if norm_Fx <= 1e-8:
                return x
            break

    if norm_Fx <= 1e-8:
        return x

    # --- Phase 2: Homotopy/Continuation Fallback ---
    if not time_up():
        result = _homotopy_fallback(F, best_x, start_time, deadline)
        if result is not None:
            Fx_res = F(result)
            if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) <= 1e-8:
                return result
            if np.linalg.norm(Fx_res) < best_norm:
                return result

    # --- Phase 3: SciPy fallback ---
    if not time_up():
        try:
            res = scipy_root(F, best_x, method='hybr', options={'maxfev': 100})
            if np.all(np.isfinite(res.x)):
                Fx_res = F(res.x)
                if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) <= 1e-8:
                    return res.x
                if np.linalg.norm(Fx_res) < best_norm:
                    return res.x
        except Exception:
            pass

    return best_x


def _homotopy_fallback(F, x_start, start_time, deadline):
    """
    Homotopy/continuation: trace path from anchor (where root is known) to target F.
    H_t(x) = (1-t)*(x - anchor) + t*F(x)
    At t=0, root is at x=anchor. At t=1, root is where F(x)=0.
    Uses Broyden's method at each t-step.
    """
    def time_up():
        return time.time() > deadline

    # Try multiple anchor points
    anchors = [x_start.copy(), x_start + 1.0, x_start - 1.0, x_start + 0.5, x_start - 0.5]
    
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
        dt = 0.2  # Initial step size in t
        
        for t_step in range(15):
            if time_up():
                break
            
            t_target = min(t_curr + dt, 1.0)
            
            # Solve H_t(x) = 0 using Broyden's method starting from x_curr
            def H_t(x_val):
                return (1 - t_target) * (x_val - anchor) + t_target * F(x_val)
            
            # Evaluate H_t at current point
            try:
                Hx = H_t(x_curr)
            except Exception:
                dt = max(dt * 0.5, 0.05)
                continue
            
            if not np.all(np.isfinite(Hx)):
                dt = max(dt * 0.5, 0.05)
                continue
            
            norm_Hx = np.linalg.norm(Hx)
            
            # Initialize Jacobian for H_t using finite differences
            h = 1e-5
            J_h = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Hp = H_t(x_curr + e)
                    Hm = H_t(x_curr - e)
                    if np.all(np.isfinite(Hp)) and np.all(np.isfinite(Hm)):
                        J_h[:, j] = (Hp - Hm) / (2 * h)
                    else:
                        Hp = H_t(x_curr + e)
                        if np.all(np.isfinite(Hp)):
                            J_h[:, j] = (Hp - Hx) / h
                        else:
                            J_h[:, j] = 0.0
                except Exception:
                    J_h[:, j] = 0.0
            
            converged = False
            x_broyden = x_curr.copy()
            Fx_broyden = Hx.copy()
            norm_broyden = norm_Hx
            
            for b_iter in range(20):
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
                
                # Line search
                alpha = 1.0
                step_ok = False
                for ls in range(10):
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
                    
                    if norm_trial < (1.0 - 1e-3 * alpha) * norm_broyden:
                        # Broyden update
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
                # Check if it's a good root of F
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

    # Try Broyden refinement from best_x found during homotopy
    if not time_up() and best_norm < np.inf:
        result = _broyden_refine(F, best_x, start_time, deadline)
        if result is not None:
            try:
                Fx_res = F(result)
                if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) <= 1e-8:
                    return result
                if np.linalg.norm(Fx_res) < best_norm:
                    return result
            except Exception:
                pass

    return best_x if best_norm < np.inf else None


def _broyden_refine(F, x0, start_time, deadline):
    """Run Broyden's method from x0 to refine to high accuracy."""
    def time_up():
        return time.time() > deadline

    x = np.array(x0, dtype=float).copy()
    try:
        Fx = F(x)
    except Exception:
        return None
    if not np.all(np.isfinite(Fx)):
        return None

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fp = F(x + e)
            Fm = F(x - e)
            if np.all(np.isfinite(Fp)) and np.all(np.isfinite(Fm)):
                J[:, j] = (Fp - Fm) / (2 * h)
            else:
                Fp = F(x + e)
                if np.all(np.isfinite(Fp)):
                    J[:, j] = (Fp - Fx) / h
                else:
                    J[:, j] = 0.0
        except Exception:
            J[:, j] = 0.0

    for iteration in range(30):
        if time_up():
            break

        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(d)):
            break
        if np.linalg.norm(d) < 1e-12:
            break

        alpha = 1.0
        step_accepted = False
        for ls in range(15):
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
            norm_new = np.linalg.norm(Fx_new)
            if norm_new <= 1e-8:
                return x_new
            if norm_new < (1.0 - 1e-3 * alpha) * norm_Fx:
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-15:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                step_accepted = True
                break
            alpha *= 0.5

        if not step_accepted:
            break

    if norm_Fx <= 1e-8:
        return x
    return x if norm_Fx < 1e-4 else None

# EVOLVE-BLOCK-END
