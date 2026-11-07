"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)

    if norm_F <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_F

    # Adaptive step size for finite differences
    h = 1e-5
    max_iter = 80
    max_ls = 5
    tol = 1e-8

    # Threshold to switch to fast polish mode
    polish_threshold = 1e-4

    total_calls = 1  # initial F(x0)

    def update_best(x_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand

    def compute_fd_jacobian(x_curr, Fx_curr):
        """Compute Jacobian via finite differences (2 calls)."""
        nonlocal total_calls
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_curr + e)
            J[:, j] = (Fx_e - Fx_curr) / h
            total_calls += 1
        return J

    def broyden_update(J, s, y):
        """Broyden's good method rank-1 update."""
        denom = s @ s
        if denom < 1e-15:
            return J
        return J + np.outer(y - J @ s, s) / denom

    # Initialize Jacobian with finite differences (2 calls)
    J = compute_fd_jacobian(x, Fx)

    # Main Broyden loop
    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        in_polish = norm_F < polish_threshold

        # Solve J dx = -Fx using Broyden approximation
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            try:
                dx = -np.linalg.solve(J + 1e-8 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                # Regularize more aggressively
                try:
                    dx = -np.linalg.solve(J + 1e-4 * np.eye(2), Fx)
                except np.linalg.LinAlgError:
                    dx = -Fx * 0.01

        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
            step_norm = np.linalg.norm(dx)

        if step_norm < 1e-15:
            # Step too small; recompute Jacobian to escape
            J = compute_fd_jacobian(x, Fx)
            if norm_F <= tol:
                return x
            continue

        if in_polish:
            # Polish mode: try full Newton step, then damped
            accepted = False
            for alpha in [1.0, 0.5, 0.25]:
                if time.time() > deadline:
                    break
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1

                if norm_candidate <= tol:
                    return x_candidate

                update_best(x_candidate, norm_candidate)

                if norm_candidate < norm_F:
                    accepted = True
                    s = x_candidate - x
                    y = F_candidate - Fx
                    J = broyden_update(J, s, y)
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break

            if not accepted:
                # Try gradient step
                x_candidate = x - 0.1 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1

                if norm_candidate < norm_F:
                    s = x_candidate - x
                    y = F_candidate - Fx
                    J = broyden_update(J, s, y)
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    update_best(x, norm_F)
                else:
                    # Recompute Jacobian to get fresh direction
                    J = compute_fd_jacobian(x, Fx)
                    break
        else:
            # Damped mode: line search
            alpha = 1.0
            accepted = False
            for ls_iter in range(max_ls):
                if time.time() > deadline:
                    break
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1

                if norm_candidate <= tol:
                    return x_candidate

                update_best(x_candidate, norm_candidate)

                if norm_candidate < norm_F * 0.5:
                    accepted = True
                    s = x_candidate - x
                    y = F_candidate - Fx
                    J = broyden_update(J, s, y)
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                # Gradient step fallback
                x_candidate = x - 0.05 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1

                if norm_candidate < norm_F:
                    s = x_candidate - x
                    y = F_candidate - Fx
                    J = broyden_update(J, s, y)
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    update_best(x, norm_F)
                else:
                    # Recompute Jacobian and break to try restarts
                    J = compute_fd_jacobian(x, Fx)
                    break

        if norm_F <= tol:
            return x

        # Periodically recompute Jacobian to prevent drift
        if iteration % 10 == 9 and not in_polish:
            J = compute_fd_jacobian(x, Fx)

        # Adaptive h
        if norm_F > 1.0:
            h = 1e-4
        elif norm_F < 1e-3:
            h = 1e-6

    # Multi-start restarts if main loop didn't converge
    if best_norm > tol:
        radii = [0.5, 1.0, 2.0, 5.0, 10.0]
        angles = [i * np.pi / 4 for i in range(8)]

        for r in radii:
            for angle in angles:
                if time.time() > deadline:
                    break
                if total_calls > 450:
                    break

                x_start = x0 + r * np.array([np.cos(angle), np.sin(angle)])
                try:
                    F_start = F(x_start)
                    total_calls += 1
                except Exception:
                    continue

                norm_start = np.linalg.norm(F_start)
                if norm_start <= tol:
                    return x_start
                update_best(x_start, norm_start)

                # Quick Broyden from this start
                x_curr = x_start.copy()
                Fx_curr = F_start.copy()
                norm_curr = norm_start

                # Init Jacobian for this restart
                try:
                    J_curr = compute_fd_jacobian(x_curr, Fx_curr)
                except Exception:
                    continue

                for _ in range(12):
                    if time.time() > deadline:
                        break
                    if total_calls > 480:
                        break

                    in_polish_local = norm_curr < polish_threshold

                    try:
                        dx = -np.linalg.solve(J_curr, Fx_curr)
                    except np.linalg.LinAlgError:
                        try:
                            dx = -np.linalg.solve(J_curr + 1e-8 * np.eye(2), Fx_curr)
                        except np.linalg.LinAlgError:
                            dx = -Fx_curr * 0.01

                    step_norm = np.linalg.norm(dx)
                    if step_norm > 1.0:
                        dx = dx / step_norm
                    if step_norm < 1e-15:
                        try:
                            J_curr = compute_fd_jacobian(x_curr, Fx_curr)
                        except Exception:
                            break
                        continue

                    if in_polish_local:
                        accepted = False
                        for alpha in [1.0, 0.5]:
                            if time.time() > deadline:
                                break
                            x_candidate = x_curr + alpha * dx
                            F_candidate = F(x_candidate)
                            norm_candidate = np.linalg.norm(F_candidate)
                            total_calls += 1

                            if norm_candidate <= tol:
                                return x_candidate
                            update_best(x_candidate, norm_candidate)

                            if norm_candidate < norm_curr:
                                accepted = True
                                s = x_candidate - x_curr
                                y = F_candidate - Fx_curr
                                J_curr = broyden_update(J_curr, s, y)
                                x_curr = x_candidate
                                Fx_curr = F_candidate
                                norm_curr = norm_candidate
                                break
                        if not accepted:
                            break
                    else:
                        alpha = 1.0
                        accepted = False
                        for _ in range(4):
                            if time.time() > deadline:
                                break
                            x_candidate = x_curr + alpha * dx
                            F_candidate = F(x_candidate)
                            norm_candidate = np.linalg.norm(F_candidate)
                            total_calls += 1

                            if norm_candidate <= tol:
                                return x_candidate
                            update_best(x_candidate, norm_candidate)

                            if norm_candidate < norm_curr * 0.5:
                                accepted = True
                                s = x_candidate - x_curr
                                y = F_candidate - Fx_curr
                                J_curr = broyden_update(J_curr, s, y)
                                x_curr = x_candidate
                                Fx_curr = F_candidate
                                norm_curr = norm_candidate
                                break
                            else:
                                alpha *= 0.5
                        if not accepted:
                            break

                    if norm_curr <= tol:
                        return x_curr

    return best_x
# EVOLVE-BLOCK-END
