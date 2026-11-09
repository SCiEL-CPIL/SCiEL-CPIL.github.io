"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses damped Newton with Broyden updates and finite-difference Jacobian.
    """
    start_time = time.time()
    time_limit = 110.0  # safety margin before 120s

    def time_left():
        return time_limit - (time.time() - start_time)

    def norm(v):
        return float(np.linalg.norm(v))

    x = np.array(x0, dtype=float)
    Fx = F(x)
    if norm(Fx) <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm(Fx)

    # Initial Jacobian via finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h

    # Damped Newton with Broyden updates
    for it in range(50):
        if time_left() < 2.0:
            break

        cur_norm = norm(Fx)
        if cur_norm <= 1e-8:
            return x
        if cur_norm < best_norm:
            best_norm = cur_norm
            best_x = x.copy()

        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Add small regularization
            J_reg = J + np.eye(2) * 1e-6
            try:
                dx = np.linalg.solve(J_reg, -Fx)
            except np.linalg.LinAlgError:
                break

        # Damped Newton: find lambda in (0,1] such that ||F(x + lambda dx)|| < ||Fx||
        lam = 1.0
        improved = False
        for _ in range(10):
            x_new = x + lam * dx
            F_new = F(x_new)
            if norm(F_new) < cur_norm * 0.9:  # sufficient decrease
                improved = True
                break
            lam *= 0.5

        if not improved:
            # Try a smaller step
            lam = 0.1
            x_new = x + lam * dx
            F_new = F(x_new)
            if norm(F_new) < cur_norm:
                improved = True
            else:
                # No improvement, break
                break

        # Broyden update
        dx_actual = x_new - x
        dF = F_new - Fx
        r = dF - J @ dx_actual
        denom = dx_actual @ dx_actual
        if denom > 1e-12:
            J = J + np.outer(r, dx_actual) / denom

        x = x_new
        Fx = F_new

    if best_norm <= 1e-8:
        return best_x

    # Fallback: try a few random restarts if time permits
    if time_left() > 10.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time_left() < 5.0:
                break
            x_try = x0 + rng.normal(0, 0.1, size=2)
            F_try = F(x_try)
            if norm(F_try) <= 1e-8:
                return x_try
            if norm(F_try) < best_norm:
                best_x = x_try.copy()
                best_norm = norm(F_try)

    return best_x

# EVOLVE-BLOCK-END
