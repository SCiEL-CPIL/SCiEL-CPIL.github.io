# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 140.0
    max_calls = 450
    calls = [0]

    def F_wrap(x):
        calls[0] += 1
        try:
            r = np.asarray(F(x), dtype=float)
        except Exception:
            return np.array([np.nan, np.nan])
        if r.shape != (2,) or not np.all(np.isfinite(r)):
            return np.array([np.nan, np.nan])
        return r

    def time_up():
        return time.time() >= deadline

    def budget_up():
        return calls[0] >= max_calls

    def stop():
        return time_up() or budget_up()

    def norm(v):
        return float(np.linalg.norm(v))

    def finite(v):
        return np.all(np.isfinite(v))

    def check_convergence(x):
        if not finite(x):
            return False
        Fx = F_wrap(x)
        if not finite(Fx):
            return False
        return norm(Fx) <= 1e-10

    # Track best point found
    best_x = np.array(x0, dtype=float)
    best_norm = np.inf
    try:
        Fx0 = F_wrap(np.array(x0, dtype=float))
        if finite(Fx0):
            best_norm = norm(Fx0)
    except Exception:
        pass

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        if finite(x) and finite(Fx):
            n = norm(Fx)
            if n < best_norm:
                best_norm = n
                best_x = x.copy()

    # --- Broyden's method with line search ---
    def broyden(x0_arr):
        x = np.array(x0_arr, dtype=float)
        Fx = F_wrap(x)
        update_best(x, Fx)
        if not finite(Fx):
            return None
        if norm(Fx) <= 1e-10:
            return x

        # Initial Jacobian via finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = F_wrap(x + e)
            if not finite(Fx_p):
                h = 1e-5
                Fx_p = F_wrap(x + e)
            if finite(Fx_p):
                J[:, j] = (Fx_p - Fx) / h
            else:
                J[:, j] = 1.0  # fallback
        if not finite(J) or np.linalg.det(J) < 1e-12:
            J = np.eye(2) * 1e-3

        s_prev = None
        y_prev = None

        for _ in range(200):
            if stop():
                return None

            try:
                d = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                d = -Fx

            if not finite(d):
                return None

            # Backtracking line search
            alpha = 1.0
            Fx_new = None
            for _ls in range(30):
                x_try = x + alpha * d
                Fx_try = F_wrap(x_try)
                if not finite(Fx_try):
                    alpha *= 0.5
                    continue
                # Armijo: ||Fx_new||^2 <= ||Fx||^2 - c*alpha*||Fx||^2
                if norm(Fx_try) <= norm(Fx) - 0.5 * alpha * norm(Fx) or norm(Fx_try) < 1e-10:
                    Fx_new = Fx_try
                    break
                alpha *= 0.5
            else:
                # Accept best step
                x_try = x + d
                Fx_try = F_wrap(x_try)
                if not finite(Fx_try):
                    return None
                Fx_new = Fx_try

            x = x + alpha * d
            Fx = Fx_new
            update_best(x, Fx)

            if norm(Fx) <= 1e-10:
                return x

            # Broyden rank-1 update
            s = alpha * d
            y = Fx - Fx_old if 'Fx_old' in dir() else None
            # Recompute: we need Fx before update
            # Actually we already have Fx (new) and we need old Fx
            # Let's restructure: save Fx before step

        return None

    # Better Broyden implementation
    def broyden2(x0_arr):
        x = np.array(x0_arr, dtype=float)
        Fx = F_wrap(x)
        update_best(x, Fx)
        if not finite(Fx):
            return None
        if norm(Fx) <= 1e-10:
            return x

        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = F_wrap(x + e)
            if not finite(Fx_p):
                h = 1e-5
                e[j] = h
                Fx_p = F_wrap(x + e)
            if finite(Fx_p):
                J[:, j] = (Fx_p - Fx) / h
            else:
                J[:, j] = 1.0
        if not finite(J):
            J = np.eye(2) * 1e-3

        for _ in range(200):
            if stop():
                return None

            Fx_old = Fx.copy()
            try:
                d = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                d = -Fx

            if not finite(d):
                return None

            alpha = 1.0
            Fx_new = None
            x_new = None
            for _ls in range(30):
                x_try = x + alpha * d
                Fx_try = F_wrap(x_try)
                if not finite(Fx_try):
                    alpha *= 0.5
                    continue
                if norm(Fx_try) <= norm(Fx_old) * 0.9 or norm(Fx_try) < 1e-10:
                    Fx_new = Fx_try
                    x_new = x_try
                    break
                alpha *= 0.5
            if Fx_new is None:
                x_try = x + d
                Fx_try = F_wrap(x_try)
                if not finite(Fx_try):
                    return None
                Fx_new = Fx_try
                x_new = x_try

            s = x_new - x
            y = Fx_new - Fx_old
            x = x_new
            Fx = Fx_new
            update_best(x, Fx)

            if norm(Fx) <= 1e-10:
                return x

            # Broyden update: J += (y - J*s) * s^T / (s^T * s)
            s_dot_s = s @ s
            if s_dot_s > 1e-16 and finite(s) and finite(y):
                J = J + np.outer(y - J @ s, s) / s_dot_s
                if not finite(J):
                    J = np.eye(2) * 1e-3

        return None

    # Damped Newton fallback
    def damped_newton(x0_arr):
        x = np.array(x0_arr, dtype=float)
        Fx = F_wrap(x)
        update_best(x, Fx)
        if not finite(Fx):
            return None
        if norm(Fx) <= 1e-10:
            return x

        for _ in range(100):
            if stop():
                return None
            h = 1e-6
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = F_wrap(x + e)
                if not finite(Fx_p):
                    return None
                J[:, j] = (Fx_p - Fx) / h
            if not finite(J):
                return None
            try:
                d = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                d = -Fx
            if not finite(d):
                return None

            alpha = 1.0
            for _ls in range(30):
                x_try = x + alpha * d
                Fx_try = F_wrap(x_try)
                if not finite(Fx_try):
                    alpha *= 0.5
                    continue
                if norm(Fx_try) < norm(Fx):
                    x = x_try
                    Fx = Fx_try
                    update_best(x, Fx)
                    break
                alpha *= 0.5
            else:
                return None

            if norm(Fx) <= 1e-10:
                return x

        return None

    # Try Broyden first
    result = broyden2(np.array(x0, dtype=float))
    if result is not None:
        return result

    # Try damped Newton
    result = damped_newton(np.array(x0, dtype=float))
    if result is not None:
        return result

    # Try random restarts
    rng = np.random.RandomState(42)
    for _ in range(20):
        if stop():
            break
        x_try = np.array(x0, dtype=float) + rng.randn(2) * 0.1
        result = broyden2(x_try)
        if result is not None:
            return result

    # Return best found
    return best_x
# EVOLVE-BLOCK-END
