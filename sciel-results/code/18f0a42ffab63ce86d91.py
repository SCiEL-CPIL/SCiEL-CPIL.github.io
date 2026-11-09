import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method for fast convergence, with Newton polishing for final precision.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def time_ok():
        return time.time() < deadline

    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 450  # Keep well under 500 to avoid penalty

    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)

    best_x = x0.copy()
    best_norm = np.inf

    def update_best(x, fx):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            fx = np.asarray(fx, dtype=float)
            if x.shape != (2,) or fx.shape != (2,):
                return
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    # Helper: Forward Difference Jacobian (2 calls, using existing Fx)
    def jac_forward(x, Fx):
        h = 1e-6 * max(1.0, np.linalg.norm(x))
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])

        Fx_p1 = F_counted(x + e1)
        Fx_p2 = F_counted(x + e2)

        if not (np.all(np.isfinite(Fx_p1)) and np.all(np.isfinite(Fx_p2)) and np.all(np.isfinite(Fx))):
            return None
        J = np.column_stack([(Fx_p1 - Fx) / h, (Fx_p2 - Fx) / h])
        return J

    # Helper: Central Difference Jacobian (4 calls)
    def jac_central(x):
        h = 1e-7 * max(1.0, np.linalg.norm(x))
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_p1 = F_counted(x + e1)
        Fx_m1 = F_counted(x - e1)
        Fx_p2 = F_counted(x + e2)
        Fx_m2 = F_counted(x - e2)

        if not (np.all(np.isfinite(Fx_p1)) and np.all(np.isfinite(Fx_m1)) and
                np.all(np.isfinite(Fx_p2)) and np.all(np.isfinite(Fx_m2))):
            return None
        J = np.column_stack([(Fx_p1 - Fx_m1) / (2*h), (Fx_p2 - Fx_m2) / (2*h)])
        return J

    # Core Broyden Solver
    def solve_broyden(x_init, max_iter=30):
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return False

        x = np.array(x_init, dtype=float)

        # 1. Initial F and Jacobian
        Fx = F_counted(x)
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= 1e-8:
            return True

        J = jac_forward(x, Fx)
        if J is None:
            return False

        # Ensure J is not singular
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-10:
            J += np.eye(2) * 1e-8

        prev_norm = np.linalg.norm(Fx)
        stagnation_count = 0

        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 5 or not time_ok():
                break

            # 2. Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break

            # 3. Damping if step is too large
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            elif step_norm < 1e-14:
                break

            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                x_new = x - dx
                if not np.all(np.isfinite(x_new)):
                    break

            Fx_new = F_counted(x_new)
            if not np.all(np.isfinite(Fx_new)):
                break

            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)

            if n_new <= 1e-8:
                return True

            # Stagnation check
            if n_new > prev_norm * 0.9:
                stagnation_count += 1
                if stagnation_count >= 3:
                    break
            else:
                stagnation_count = 0
            prev_norm = n_new

            # 4. Broyden Update (Good Method)
            s = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s, s)
            if denom > 1e-20:
                J = J + np.outer(y - J @ s, s) / denom

            x = x_new
            Fx = F_new

        return best_norm <= 1e-8

    # Newton Polisher (High precision)
    def polish_newton(x_init, max_iter=15):
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS - 10 or not time_ok():
            return False

        x = np.array(x_init, dtype=float)

        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 2 or not time_ok():
                break

            Fx = F_counted(x)
            update_best(x, Fx)
            n = np.linalg.norm(Fx)

            if n <= 1e-8:
                return True

            if n > 1e-3:
                break

            # Use central difference for higher precision
            J = jac_central(x)
            if J is None:
                break

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break

            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break

            Fx_new = F_counted(x_new)
            update_best(x_new, Fx_new)
            n_new = np.linalg.norm(Fx_new)

            if n_new < n:
                x = x_new
            else:
                x_half = x + 0.5 * dx
                Fx_half = F_counted(x_half)
                update_best(x_half, Fx_half)
                if np.linalg.norm(Fx_half) < n:
                    x = x_half
                else:
                    break

        return best_norm <= 1e-8

    # --- Execution Plan ---

    # 1. Try Broyden from x0
    if solve_broyden(x0):
        return best_x

    # 2. If close, try to polish
    if best_norm < 1e-2 and call_count < MAX_CALLS - 10 and time_ok():
        if polish_newton(best_x):
            return best_x

    # 3. Restart with perturbations if not converged
    scale = max(1.0, np.linalg.norm(x0))
    restarts = [
        x0 + np.array([scale, 0.0]),
        x0 + np.array([0.0, scale]),
        x0 + np.array([scale, scale]) * 0.5,
        x0 * 0.5,
        np.array([0.0, 0.0]),
        x0 + np.array([scale, -scale]) * 0.5,
        x0 - np.array([scale, 0.0]),
        x0 - np.array([0.0, scale]),
    ]

    for r in restarts:
        if call_count >= MAX_CALLS - 25 or not time_ok():
            break
        if not np.all(np.isfinite(r)):
            continue

        if solve_broyden(r):
            return best_x

        if best_norm < 1e-2 and call_count < MAX_CALLS - 10 and time_ok():
            if polish_newton(best_x):
                return best_x

    # 4. Final desperate polish if we are close
    if best_norm < 1e-1 and call_count < MAX_CALLS - 5 and time_ok():
        if polish_newton(best_x):
            return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
