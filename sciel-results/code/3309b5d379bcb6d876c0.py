"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using a robust hybrid approach:
    1. Levenberg-Marquardt with Broyden rank-1 Jacobian updates.
    2. Adaptive damping based on residual reduction.
    3. Time-guarded: return best result before deadline.
    
    Budget: <500 calls to F.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 135.0  # safety margin
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None  # signal to stop
        if call_count >= 498:  # leave 2 calls for safety
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

    # Compute initial Jacobian via finite differences
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = _call_F(x + e)
        if Fx_e is None:
            break
        J[:, j] = (Fx_e - Fx) / h

    # LM parameter: start moderate
    mu = 1e-3

    # LM iterations with Broyden updates
    for it in range(200):
        if time.time() > deadline:
            break
        if call_count >= 495:
            break

        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Solve (J^T J + mu * ||J||^2 I) * dx = -J^T Fx
        JtJ = J.T @ J
        JtF = J.T @ Fx
        # Regularization: mu * scale * I where scale = trace(JtJ)/2 for 2x2
        scale = np.trace(JtJ) / 2.0
        if scale < 1e-12:
            scale = 1.0
        A = JtJ + mu * scale * np.eye(2)
        b = -JtF

        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            dx = np.linalg.pinv(A) @ b

        # Limit step size
        norm_dx = np.linalg.norm(dx)
        max_step = 10.0
        if norm_dx > max_step:
            dx = dx / norm_dx * max_step

        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            x_new = x + dx * 0.1
            if not np.all(np.isfinite(x_new)):
                break

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        # Accept or reject step based on residual reduction
        if norm_Fx_new < norm_Fx:
            # Accept step, decrease mu
            mu = max(mu * 0.1, 1e-10)
            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            if np.linalg.norm(s) > 1e-15:
                J = J + np.outer(y - J @ s, s.T) / (s.T @ s)
            x = x_new
            Fx = Fx_new
            if norm_Fx_new <= 1e-8:
                return x
        else:
            # Reject step, increase mu
            mu = min(mu * 10.0, 1e10)
            # If mu is very large, we're basically doing gradient descent
            # Try a smaller step in the negative gradient direction
            if mu > 1e6:
                # Gradient of 0.5*||F||^2 is J^T F
                grad = J.T @ Fx
                norm_grad = np.linalg.norm(grad)
                if norm_grad > 1e-15:
                    step = -grad / norm_grad * 0.01
                    x_new2 = x + step
                    if np.all(np.isfinite(x_new2)):
                        Fx_new2 = _call_F(x_new2)
                        if Fx_new2 is not None:
                            norm_new2 = np.linalg.norm(Fx_new2)
                            if norm_new2 < norm_Fx:
                                x = x_new2
                                Fx = Fx_new2
                                if norm_new2 < best_norm:
                                    best_x = x.copy()
                                    best_norm = norm_new2
                                mu = 1e-3
                                continue

    # Final polish: if close, do a few more Newton steps
    if best_norm < 1e-4:
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            for _ in range(10):
                if time.time() > deadline or call_count >= 498:
                    break
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx <= 1e-8:
                    return x
                # Recompute Jacobian for polish
                h = 1e-8
                J_polish = np.zeros((2, 2))
                Fx_ref = _call_F(x)
                if Fx_ref is None:
                    break
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_e = _call_F(x + e)
                    if Fx_e is None:
                        break
                    J_polish[:, j] = (Fx_e - Fx_ref) / h
                try:
                    dx = np.linalg.solve(J_polish, -Fx)
                    x_new = x + dx
                    if np.all(np.isfinite(x_new)):
                        Fx_new = _call_F(x_new)
                        if Fx_new is not None:
                            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                                x = x_new
                                Fx = Fx_new
                                if np.linalg.norm(Fx) < best_norm:
                                    best_x = x.copy()
                                    best_norm = np.linalg.norm(Fx)
                except np.linalg.LinAlgError:
                    break

    return best_x
# EVOLVE-BLOCK-END
