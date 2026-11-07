"""Root finder for 3 fixed two-dimensional systems using adaptive multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses an adaptive multi-strategy approach:
    1. Check initial point
    2. Broyden's method with central-difference Jacobian
    3. Damped Newton with fresh Jacobian and multiple restarts
    4. Global search via random restarts + local refinement
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    n = 2
    tol = 1e-8

    def elapsed_ok():
        return time.time() < deadline

    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = float(np.linalg.norm(Fx))

    if norm_Fx <= tol:
        return x

    best_x = x.copy()
    best_norm = norm_Fx

    def update_best(x_cand, Fx_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand
            return True
        return False

    # Strategy 1: Broyden's method with central-difference Jacobian initialization
    if elapsed_ok():
        h = 1e-5
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            Fm = np.array(F(x - e), dtype=float)
            J[:, j] = (Fp - Fm) / (2 * h)

        max_iter_broyden = 40
        for it in range(max_iter_broyden):
            if not elapsed_ok():
                break

            # Try to solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break

            # Line search to ensure sufficient decrease
            alpha = 1.0
            max_ls = 12
            ls_success = False
            for _ in range(max_ls):
                x_new = x + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                norm_Fx_new = float(np.linalg.norm(Fx_new))

                if norm_Fx_new <= tol:
                    return x_new

                update_best(x_new, Fx_new, norm_Fx_new)

                # Sufficient decrease condition
                if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                    # Good-Broyden update
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

            if norm_Fx <= tol:
                return x

    # Strategy 2: Damped Newton with fresh Jacobian, multiple restarts
    if elapsed_ok():
        for restart in range(12):
            if not elapsed_ok():
                break

            # Recompute Jacobian with central differences
            Fx = np.array(F(x), dtype=float)
            norm_Fx = float(np.linalg.norm(Fx))
            if norm_Fx <= tol:
                return x
            update_best(x, Fx, norm_Fx)

            # Adaptive step size for Jacobian estimation
            h_jac = 1e-5
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = h_jac
                Fp = np.array(F(x + e), dtype=float)
                Fm = np.array(F(x - e), dtype=float)
                J[:, j] = (Fp - Fm) / (2 * h_jac)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    # Perturb x slightly and try again
                    x = x + np.random.normal(0, 0.1, n)
                    continue

            # Damped Newton step
            alpha = 1.0
            max_ls = 15
            newton_success = False
            for _ in range(max_ls):
                x_new = x + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                norm_Fx_new = float(np.linalg.norm(Fx_new))

                if norm_Fx_new <= tol:
                    return x_new

                update_best(x_new, Fx_new, norm_Fx_new)

                # Armijo condition for damping
                if norm_Fx_new < norm_Fx - 1e-4 * alpha * float(np.dot(dx, J.T @ dx)):
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    newton_success = True
                    break

                alpha *= 0.5

            if not newton_success:
                # If Newton step failed, perturb x to escape local issues
                x = x + np.random.normal(0, 0.05, n)

    # Strategy 3: Random restarts with local refinement
    if elapsed_ok() and best_norm > tol:
        # Try multiple random starting points with local Broyden/Newton
        for attempt in range(15):
            if not elapsed_ok():
                break

            # Generate a random starting point in a reasonable range
            scale = max(1.0, np.linalg.norm(best_x))
            x_try = best_x + np.random.normal(0, scale, n)
            Fx_try = np.array(F(x_try), dtype=float)
            norm_try = float(np.linalg.norm(Fx_try))

            if norm_try <= tol:
                return x_try

            update_best(x_try, Fx_try, norm_try)

            # Quick local refinement from this point
            if elapsed_ok():
                h = 1e-5
                J = np.zeros((n, n))
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = h
                    Fp = np.array(F(x_try + e), dtype=float)
                    Fm = np.array(F(x_try - e), dtype=float)
                    J[:, j] = (Fp - Fm) / (2 * h)

                for it in range(10):
                    if not elapsed_ok():
                        break

                    try:
                        dx = np.linalg.solve(J, -Fx_try)
                    except np.linalg.LinAlgError:
                        try:
                            dx = np.linalg.lstsq(J, -Fx_try, rcond=None)[0]
                        except:
                            break

                    alpha = 1.0
                    ls_success = False
                    for _ in range(8):
                        x_new = x_try + alpha * dx
                        Fx_new = np.array(F(x_new), dtype=float)
                        norm_Fx_new = float(np.linalg.norm(Fx_new))

                        if norm_Fx_new <= tol:
                            return x_new

                        update_best(x_new, Fx_new, norm_Fx_new)

                        if norm_Fx_new < (1 - 1e-4 * alpha) * norm_try:
                            s = x_new - x_try
                            y = Fx_new - Fx_try
                            s_dot_s = float(np.dot(s, s))
                            if s_dot_s > 1e-30:
                                J += np.outer(y - J @ s, s) / s_dot_s
                            x_try = x_new
                            Fx_try = Fx_new
                            norm_try = norm_Fx_new
                            ls_success = True
                            break

                        alpha *= 0.5

                    if not ls_success:
                        break

                    if norm_try <= tol:
                        return x_try

    # Strategy 4: Final grid search around best_x if still not converged
    if elapsed_ok() and best_norm > tol:
        offsets = [
            [0.01, 0.0], [-0.01, 0.0], [0.0, 0.01], [0.0, -0.01],
            [0.001, 0.0], [-0.001, 0.0], [0.0, 0.001], [0.0, -0.001],
            [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1],
            [0.05, 0.05], [-0.05, 0.05], [0.05, -0.05], [-0.05, -0.05],
        ]
        for off in offsets:
            if not elapsed_ok():
                break
            x_try = best_x + np.array(off)
            Fx_try = np.array(F(x_try), dtype=float)
            norm_try = float(np.linalg.norm(Fx_try))
            if norm_try <= tol:
                return x_try
            if norm_try < best_norm:
                best_x = x_try
                best_norm = norm_try

    return best_x
# EVOLVE-BLOCK-END
