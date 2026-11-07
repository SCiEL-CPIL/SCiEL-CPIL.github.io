"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with adaptive Jacobian re-anchoring.
    2. Line search for robustness.
    3. Final polish with Newton's method when close to root.
    4. Adaptive restarts from promising points.
    """
    start_time = time.time()
    time_limit = 145.0

    def time_ok():
        return (time.time() - start_time) < time_limit

    call_count = [0]

    def safe_eval_F(x):
        try:
            x_arr = np.array(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                call_count[0] += 1
                return None
            result = np.array(F(x_arr), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None

    def compute_jacobian_fd(x, Fx, h=1e-7):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J

    def compute_jacobian_central(x, h=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J

    def newton_polish(x_start, Fx_start, max_iter=15, call_budget=480):
        x = np.array(x_start, dtype=float).copy()
        Fx = np.array(Fx_start, dtype=float).copy()

        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break

            J = compute_jacobian_central(x, h=1e-6)
            if J is None:
                J = compute_jacobian_fd(x, Fx, h=1e-6)
                if J is None:
                    break

            if abs(np.linalg.det(J)) < 1e-12:
                break

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break

            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True

            x = x_new
            Fx = Fx_new

        return x, Fx, (np.linalg.norm(Fx) <= 1e-8) if Fx is not None else False

    def broyden_direct(x_start, max_iter=40, call_budget=450):
        x = np.array(x_start, dtype=float).copy()
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True

        J = compute_jacobian_fd(x, Fx, h=1e-7)
        if J is None:
            J = compute_jacobian_fd(x, Fx, h=1e-5)
            if J is None:
                return x, Fx, False

        if abs(np.linalg.det(J)) < 1e-12:
            return x, Fx, False

        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)

        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J_pert = J + 1e-8 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_pert, -Fx)
                except np.linalg.LinAlgError:
                    break

            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)

            alpha = 1.0
            found_step = False
            for _ in range(8):
                x_new = x + alpha * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    alpha *= 0.5
                    continue

                Fx_new_norm = np.linalg.norm(Fx_new)
                if Fx_new_norm < best_norm:
                    found_step = True
                    break
                alpha *= 0.5

            if not found_step:
                break

            x_new = x + alpha * dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break

            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm

            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True

            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s

            # Re-anchor Jacobian periodically
            if (it + 1) % 10 == 0:
                J_new = compute_jacobian_fd(x_new, Fx_new, h=1e-7)
                if J_new is not None:
                    J = J_new

            x = x_new
            Fx = Fx_new

        return best_x, best_Fx, best_norm <= 1e-8

    # First: try direct Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_direct(x0_arr, max_iter=40, call_budget=400)
    if success:
        return x

    best_x = x.copy() if Fx is not None else x0_arr.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf

    # Try Newton polish on best point
    if Fx is not None and best_norm < 1e-4:
        x_polish, Fx_polish, success_polish = newton_polish(best_x, Fx, max_iter=10, call_budget=450)
        if success_polish:
            return x_polish
        if Fx_polish is not None:
            norm_polish = np.linalg.norm(Fx_polish)
            if norm_polish < best_norm:
                best_norm = norm_polish
                best_x = x_polish.copy()

    # Generate diverse restart points
    n0_norm = np.linalg.norm(x0_arr)
    if n0_norm < 1e-10:
        n0_norm = 1.0

    targets = [x0_arr.copy()]

    # Scaled versions
    for scale in [0.5, 2.0, -1.0, 0.2, 5.0, 0.1, -2.0, 10.0]:
        targets.append(scale * x0_arr)

    # Perpendicular directions
    perp = np.array([-x0_arr[1], x0_arr[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-10:
        perp = perp / perp_norm * n0_norm
        targets.append(x0_arr + perp)
        targets.append(x0_arr - perp)
        targets.append(perp)
        targets.append(-perp)

    # Origin
    targets.append(np.zeros(2))

    # Axis-aligned
    targets.append(np.array([n0_norm, 0.0]))
    targets.append(np.array([-n0_norm, 0.0]))
    targets.append(np.array([0.0, n0_norm]))
    targets.append(np.array([0.0, -n0_norm]))

    # Diagonal
    targets.append(np.array([n0_norm / np.sqrt(2), n0_norm / np.sqrt(2)]))
    targets.append(np.array([-n0_norm / np.sqrt(2), -n0_norm / np.sqrt(2)]))
    targets.append(np.array([n0_norm / np.sqrt(2), -n0_norm / np.sqrt(2)]))
    targets.append(np.array([-n0_norm / np.sqrt(2), n0_norm / np.sqrt(2)]))

    # Random
    rng = np.random.RandomState(42)
    for _ in range(8):
        targets.append(rng.randn(2) * max(1.0, n0_norm))

    # Sort targets by distance to x0 (try closer ones first)
    target_dists = [np.linalg.norm(t - x0_arr) for t in targets]
    sorted_indices = np.argsort(target_dists)
    targets = [targets[i] for i in sorted_indices]

    # Try Broyden from each target
    for target in targets:
        if not time_ok() or call_count[0] >= 470:
            break

        x_b, Fx_b, success_b = broyden_direct(target, max_iter=30, call_budget=470)
        if success_b:
            return x_b

        if Fx_b is not None:
            norm_b = np.linalg.norm(Fx_b)
            if norm_b < best_norm:
                best_norm = norm_b
                best_x = x_b.copy()

                # Try polishing if close
                if norm_b < 1e-4 and call_count[0] < 480:
                    x_polish, Fx_polish, success_polish = newton_polish(best_x, Fx_b, max_iter=8, call_budget=480)
                    if success_polish:
                        return x_polish
                    if Fx_polish is not None:
                        norm_polish = np.linalg.norm(Fx_polish)
                        if norm_polish < best_norm:
                            best_norm = norm_polish
                            best_x = x_polish.copy()

    # Final polish attempt
    if best_norm < 1e-3 and call_count[0] < 490:
        x_polish, Fx_polish, success_polish = newton_polish(best_x, None, max_iter=10, call_budget=490)
        if success_polish:
            return x_polish
        if Fx_polish is not None:
            norm_polish = np.linalg.norm(Fx_polish)
            if norm_polish < best_norm:
                best_norm = norm_polish
                best_x = x_polish.copy()

    return best_x
