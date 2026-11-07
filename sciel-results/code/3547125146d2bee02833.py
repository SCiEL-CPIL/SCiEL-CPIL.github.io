"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method (good for small n) with line search safeguards
    and a deadline guard to avoid timeouts.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-3, -1e-3])
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x

    # Check if already at root
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Initialize Jacobian using finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-6 * max(1.0, abs(x[j]))
            Fx_pert = F(x + e)
            J[:, j] = (Fx_pert - Fx) / e[j]
        return J

    J = compute_jacobian(x, Fx)

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    for iteration in range(100):
        if time.time() > deadline:
            break

        # Check convergence
        if best_norm <= 1e-8:
            return best_x

        # Try to solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Reset Jacobian
            J = compute_jacobian(x, Fx)
            try:
                dx = -np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                break

        # Line search with backtracking
        alpha = 1.0
        Fx_old = Fx.copy()
        norm_old = np.linalg.norm(Fx)

        for ls_iter in range(20):
            x_trial = x + alpha * dx
            Fx_trial = F(x_trial)
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            norm_trial = np.linalg.norm(Fx_trial)

            # Accept if we improved
            if norm_trial < norm_old:
                x = x_trial
                Fx = Fx_trial
                best_x = x.copy()
                best_norm = norm_trial
                if best_norm <= 1e-8:
                    return x

                # Update Jacobian (Broyden update)
                s = x - (x - alpha * dx)  # = alpha * dx
                y = Fx - Fx_old
                J = J + np.outer(y - J @ s, s) / (s @ s)
                break
            else:
                alpha *= 0.5

        else:
            # No improvement in line search, try a different direction
            # Use steepest descent direction
            grad = J.T @ Fx
            if np.linalg.norm(grad) > 1e-15:
                dx = -grad / np.linalg.norm(grad)
                alpha = 1.0
                for ls_iter in range(10):
                    x_trial = x + alpha * dx
                    Fx_trial = F(x_trial)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    norm_trial = np.linalg.norm(Fx_trial)
                    if norm_trial < norm_old:
                        x = x_trial
                        Fx = Fx_trial
                        best_x = x.copy()
                        best_norm = norm_trial
                        if best_norm <= 1e-8:
                            return x
                        # Reset Jacobian
                        J = compute_jacobian(x, Fx)
                        break
                    alpha *= 0.5
                if norm_trial >= norm_old:
                    break
            else:
                break

    # If we didn't converge, try a few more Newton steps with fresh Jacobians
    for _ in range(20):
        if time.time() > deadline:
            break
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            break
        norm = np.linalg.norm(Fx)
        if norm <= 1e-8:
            return x
        if norm < best_norm:
            best_x = x.copy()
            best_norm = norm
        J = compute_jacobian(x, Fx)
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        x = x + dx

    return best_x
# EVOLVE-BLOCK-END
