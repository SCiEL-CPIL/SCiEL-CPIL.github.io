"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using Broyden's method with:
    1. Initial finite-difference Jacobian (3 calls)
    2. Broyden's good method (1 call/iteration) with adaptive damping
    3. Periodic Jacobian refresh to maintain accuracy
    4. Final high-precision polish pass
    
    Budget: <500 calls to F.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 140.0  # safety margin before 150s limit
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None  # signal to stop
        if call_count >= 495:  # hard budget limit
            return None
        call_count += 1
        result = F(x_val)
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            return None
        return result

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian using central differences for better accuracy
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        Fx_minus = _call_F(x - e)
        if Fx_plus is None or Fx_minus is None:
            # Fallback to forward difference
            Fx_e = _call_F(x + e)
            if Fx_e is None:
                break
            J[:, j] = (Fx_e - Fx) / h
        else:
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)

    # Broyden's good method iterations
    for it in range(450):  # up to 450 iterations, but budget-limited
        if time.time() > deadline:
            break
        if call_count >= 490:  # leave room for final polish
            break

        # Evaluate F at current x
        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Periodically refresh Jacobian to maintain accuracy
        if it > 0 and it % 20 == 0:
            h = 1e-5
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = _call_F(x + e)
                Fx_minus = _call_F(x - e)
                if Fx_plus is None or Fx_minus is None:
                    break
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
            continue  # skip Broyden update this iteration

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if J is singular
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except:
                break

        # Adaptive damping: start with full step, reduce if norm doesn't decrease
        norm_dx = np.linalg.norm(dx)
        if norm_dx == 0:
            break
        
        # Cap step size to avoid divergence
        max_step = 10.0
        if norm_dx > max_step:
            dx = dx / norm_dx * max_step
            norm_dx = max_step

        # Try full step first
        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            dx = dx * 0.5
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        norm_Fx_new = np.linalg.norm(Fx_new)

        # If norm increased significantly, try smaller steps
        if norm_Fx_new > norm_Fx * 1.5:
            # Backtrack line search
            alpha = 0.5
            accepted = False
            for _ in range(5):
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                Fx_new = _call_F(x_new)
                if Fx_new is None:
                    break
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < norm_Fx:
                    accepted = True
                    break
                alpha *= 0.5
            
            if not accepted or Fx_new is None:
                # Keep best_x, don't update J
                continue
            s = x_new - x
            y = Fx_new - Fx
        else:
            s = x_new - x
            y = Fx_new - Fx

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        # Broyden rank-1 update (good method)
        if np.linalg.norm(s) > 1e-15:
            J = J + np.outer(y - J @ s, s.T) / (s.T @ s)

        x = x_new
        Fx = Fx_new
        if norm_Fx_new <= 1e-8:
            return x

    # Final high-precision polish pass
    if best_norm < 1e-4 and call_count < 495:
        x_polish = best_x.copy()
        Fx_polish = _call_F(x_polish)
        if Fx_polish is not None:
            norm_polish = np.linalg.norm(Fx_polish)
            # Use more precise finite differences
            J_polish = np.zeros((2, 2))
            h_polish = 1e-7
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_polish
                Fx_plus = _call_F(x_polish + e)
                Fx_minus = _call_F(x_polish - e)
                if Fx_plus is not None and Fx_minus is not None:
                    J_polish[:, j] = (Fx_plus - Fx_minus) / (2 * h_polish)

            # A few Newton steps with precise Jacobian
            for _ in range(3):
                if call_count >= 495:
                    break
                try:
                    dx = np.linalg.solve(J_polish, -Fx_polish)
                    x_new = x_polish + dx
                    if np.all(np.isfinite(x_new)):
                        Fx_new = _call_F(x_new)
                        if Fx_new is not None:
                            norm_new = np.linalg.norm(Fx_new)
                            if norm_new < norm_polish:
                                x_polish = x_new
                                Fx_polish = Fx_new
                                norm_polish = norm_new
                                if norm_polish < best_norm:
                                    best_x = x_polish.copy()
                                    best_norm = norm_polish
                            if norm_polish <= 1e-8:
                                return x_polish
                except:
                    break

    return best_x
# EVOLVE-BLOCK-END
