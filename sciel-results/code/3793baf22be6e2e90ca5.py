"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method with safeguards.
    Falls back to damped Newton / line search if needed.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    n = 2

    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)

    if norm_Fx <= 1e-8:
        return x

    # Initialize Jacobian using finite differences
    J = np.zeros((2, 2))
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = np.array(F(x + e), dtype=float)
        J[:, j] = (Fx_e - Fx) / h

    best_x = x.copy()
    best_norm = norm_Fx

    # Broyden's method loop
    max_iter = 200
    for it in range(max_iter):
        # Check time
        if time.time() > deadline - 5:
            break

        # Solve for step
        try:
            s = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # If J is singular, use least squares
            try:
                s = np.linalg.lstsq(J, Fx, rcond=None)[0]
            except:
                break

        # Check for non-finite step
        if not np.all(np.isfinite(s)):
            break

        # Limit step size for stability
        s_norm = np.linalg.norm(s)
        if s_norm > 1.0:
            s = s / s_norm

        # Line search to ensure decrease in ||F(x)||
        alpha = 1.0
        max_ls = 30
        for ls in range(max_ls):
            x_new = x - alpha * s
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = np.linalg.norm(Fx_new)

            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            if norm_Fx_new <= 1e-8:
                return x_new

            # Check for sufficient decrease (Armijo-like)
            if norm_Fx_new < norm_Fx * (1 - 1e-4 * alpha):
                break

            alpha *= 0.5

        else:
            # If no good step found, try smaller step
            x_new = x - alpha * s
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = np.linalg.norm(Fx_new)

            if not np.all(np.isfinite(Fx_new)):
                break

        # Update best
        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        # Broyden rank-1 update
        Fx_old = Fx
        Fx = Fx_new
        norm_Fx = norm_Fx_new

        # Check convergence
        if norm_Fx <= 1e-8:
            return x_new

        # Update Jacobian using Broyden's formula
        s_actual = x_new - x
        y = Fx - Fx_old
        x = x_new

        # Broyden update: J_new = J + (y - J*s) * s^T / (s^T * s)
        s_dot_s = np.dot(s_actual, s_actual)
        if s_dot_s > 1e-16:
            J = J + np.outer(y - J @ s_actual, s_actual) / s_dot_s

        # Periodically recompute Jacobian to prevent drift
        if it % 10 == 9:
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = np.array(F(x + e), dtype=float)
                J[:, j] = (Fx_e - Fx) / h

    # If we didn't converge, try a few more Newton steps with fresh Jacobian
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        Fx = np.array(F(x), dtype=float)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x

        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = np.array(F(x + e), dtype=float)
            J[:, j] = (Fx_e - Fx) / h

        try:
            s = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(s)):
            break

        x = x - s

    # Return best result found
    if best_norm < np.linalg.norm(np.array(F(x), dtype=float)):
        return best_x
    return x
# EVOLVE-BLOCK-END
