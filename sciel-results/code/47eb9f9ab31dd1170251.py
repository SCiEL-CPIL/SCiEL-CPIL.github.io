"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method with line search.
    Falls back to damped Newton if Broyden stalls.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    n = 2

    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = float(np.linalg.norm(Fx))

    if norm_Fx <= 1e-8:
        return x

    # Initialize Jacobian estimate via secant (2 calls)
    # Use central differences for better accuracy
    h = 1e-5
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        Fp = np.array(F(x + e), dtype=float)
        Fm = np.array(F(x - e), dtype=float)
        J[:, j] = (Fp - Fm) / (2 * h)

    # Broyden's method with good-Broyden rank-1 update
    best_x = x.copy()
    best_norm = norm_Fx

    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use least squares
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Line search to ensure sufficient decrease
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

            # Sufficient decrease condition
            if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                # Good-Broyden update
                s = x_new - x
                y = Fx_new - Fx
                # Rank-1 update: J += (y - J*s) * s^T / (s^T * s)
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
            # Line search failed; try a smaller step or break
            # Try one more with very small step
            x_new = x + 1e-8 * dx
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = float(np.linalg.norm(Fx_new))
            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
            break

        if norm_Fx <= 1e-8:
            return x

    # If Broyden didn't converge, try a few more with fresh Jacobian
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Recompute Jacobian
            Fx = np.array(F(x), dtype=float)
            norm_Fx = float(np.linalg.norm(Fx))
            if norm_Fx <= 1e-8:
                return x
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx

            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-5
                Fp = np.array(F(x + e), dtype=float)
                Fm = np.array(F(x - e), dtype=float)
                J[:, j] = (Fp - Fm) / (2 * 1e-5)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

            alpha = 1.0
            for _ in range(10):
                x_new = x + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                norm_Fx_new = float(np.linalg.norm(Fx_new))
                if norm_Fx_new <= 1e-8:
                    return x_new
                if norm_Fx_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_Fx_new
                if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    break
                alpha *= 0.5

    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
