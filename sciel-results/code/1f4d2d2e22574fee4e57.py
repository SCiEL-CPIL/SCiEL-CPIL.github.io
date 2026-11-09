"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 115.0

    def time_left():
        return time_limit - (time.time() - start_time)

    def norm(v):
        return float(np.linalg.norm(v))

    def is_good(v):
        return norm(v) <= 1e-8

    def newton_damped(F, x, max_iter=80):
        """Damped Newton with finite-difference Jacobian and backtracking."""
        x = np.array(x, dtype=float)
        Fx = F(x)
        if is_good(Fx):
            return x, Fx

        best_x = x.copy()
        best_F = norm(Fx)

        for it in range(max_iter):
            if time_left() < 2.0:
                break

            # Compute finite-difference Jacobian
            h = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (F(x + e) - Fx) / h

            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try pseudo-inverse
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            # Backtracking line search
            alpha = 1.0
            for _ in range(20):
                x_trial = x + alpha * dx
                Fx_trial = F(x_trial)
                if norm(Fx_trial) < best_F:
                    best_x = x_trial.copy()
                    best_F = norm(Fx_trial)
                if is_good(Fx_trial):
                    return x_trial, Fx_trial
                if norm(Fx_trial) < 0.5 * best_F:
                    x = x_trial
                    Fx = Fx_trial
                    break
                alpha *= 0.5
            else:
                # No improvement found; accept the best trial
                if best_F < norm(Fx):
                    x = best_x
                    Fx = F(best_x)
                break

            if is_good(Fx):
                return x, Fx

        return x, Fx

    # Try Newton from initial point
    x, Fx = newton_damped(F, x0, max_iter=60)
    if is_good(Fx):
        return x

    # If Newton failed, try a few random restarts with Newton
    best_x = x.copy()
    best_F = norm(Fx)
    rng = np.random.RandomState(42)

    for _ in range(30):
        if time_left() < 5.0:
            break
        # Random restart near x0 or in a wider box
        scale = 1.0 + rng.rand() * 5.0
        x_trial = x0 + scale * rng.randn(2)
        x_r, Fx_r = newton_damped(F, x_trial, max_iter=40)
        Fx_r_norm = norm(Fx_r)
        if Fx_r_norm < best_F:
            best_x = x_r.copy()
            best_F = Fx_r_norm
            if is_good(Fx_r):
                return x_r

    # Final polish: if we have a good candidate, do a tight Newton
    if best_F < 1e-4:
        x_polish, Fx_polish = newton_damped(F, best_x, max_iter=50)
        if norm(Fx_polish) < best_F:
            best_x = x_polish
            best_F = norm(Fx_polish)

    if is_good(best_F):
        return best_x

    # Return the best we found
    return best_x
# EVOLVE-BLOCK-END
