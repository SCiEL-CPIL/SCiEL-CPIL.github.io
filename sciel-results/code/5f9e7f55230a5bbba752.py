"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Pure Levenberg-Marquardt root finder for 2D systems.
    Uses Marquardt's original lambda update strategy with adaptive finite differences.
    """
    start_time = time.time()
    DEADLINE = 145.0

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    def finite_diff_jacobian(x_current, h):
        Fx = compute_F(x_current)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (compute_F(x_current + e) - Fx) / h
        return J, Fx

    # Pure Levenberg-Marquardt
    max_iters = 200
    h_base = 1e-6
    lambda_lq = 1e-4
    lambda_min = 1e-12
    lambda_max = 1e12

    for it in range(max_iters):
        if not time_ok():
            break

        Fx = compute_F(x)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()
        if norm_Fx <= 1e-10:
            break

        # Adaptive step size for finite differences
        h = h_base * max(1.0, np.linalg.norm(x)) ** 0.5
        J, Fx = finite_diff_jacobian(x, h)

        # Compute J^T J and J^T F
        JtJ = J.T @ J
        JtF = J.T @ Fx

        # Try to solve with current lambda
        found_step = False
        for _ in range(30):
            try:
                A = JtJ + lambda_lq * np.eye(2)
                dx = np.linalg.solve(A, -JtF)
            except np.linalg.LinAlgError:
                lambda_lq = min(lambda_lq * 10.0, lambda_max)
                continue

            x_trial = x + dx
            Fx_trial = compute_F(x_trial)
            norm_trial = np.linalg.norm(Fx_trial)

            if norm_trial < norm_Fx:
                # Successful step: accept and decrease lambda
                x = x_trial
                lambda_lq = max(lambda_lq * 0.1, lambda_min)
                found_step = True
                break
            else:
                # Unsuccessful: increase lambda
                lambda_lq = min(lambda_lq * 10.0, lambda_max)

        if not found_step and lambda_lq >= lambda_max:
            # Lambda too large, break
            break

    # If LM didn't converge, try a few random restarts
    if best_norm > 1e-8 and time_ok():
        for attempt in range(15):
            if not time_ok():
                break
            # Random perturbation with varying scale
            scale = 0.1 * (10 ** (attempt % 4))  # 0.1, 1, 10, 100
            x_try = best_x + np.random.randn(2) * scale
            x_cur = x_try.copy()
            lambda_cur = 1e-4

            for it in range(80):
                if not time_ok():
                    break
                Fx = compute_F(x_cur)
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_cur.copy()
                if norm_Fx <= 1e-10:
                    break

                h = h_base * max(1.0, np.linalg.norm(x_cur)) ** 0.5
                J, Fx = finite_diff_jacobian(x_cur, h)
                JtJ = J.T @ J
                JtF = J.T @ Fx

                found_step = False
                for _ in range(20):
                    try:
                        A = JtJ + lambda_cur * np.eye(2)
                        dx = np.linalg.solve(A, -JtF)
                    except np.linalg.LinAlgError:
                        lambda_cur = min(lambda_cur * 10.0, lambda_max)
                        continue

                    x_trial = x_cur + dx
                    Fx_trial = compute_F(x_trial)
                    norm_trial = np.linalg.norm(Fx_trial)

                    if norm_trial < norm_Fx:
                        x_cur = x_trial
                        lambda_cur = max(lambda_cur * 0.1, lambda_min)
                        found_step = True
                        break
                    else:
                        lambda_cur = min(lambda_cur * 10.0, lambda_max)

                if not found_step and lambda_cur >= lambda_max:
                    break

    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
