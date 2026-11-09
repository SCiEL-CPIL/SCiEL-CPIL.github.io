"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 115.0

    def time_left():
        return time_limit - (time.time() - start_time)

    def norm(v):
        return np.linalg.norm(v)

    def broyden(F, x, max_iter=100):
        x = np.array(x, dtype=float)
        Fx = F(x)
        if norm(Fx) <= 1e-8:
            return x, Fx

        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h

        for _ in range(max_iter):
            if time_left() < 1.0:
                break

            try:
                dx = -np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                dx = -Fx
                dx /= max(norm(dx), 1e-15)

            alpha = 1.0
            Fx_new = None
            for _ in range(20):
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                if norm(Fx_new) < norm(Fx) * (1 - 1e-4 * alpha):
                    break
                alpha *= 0.5

            if Fx_new is None:
                break

            s = x_new - x
            y = Fx_new - Fx
            J = J + (np.outer(y, s.T) - np.outer(J @ s, s.T)) / (s @ s + 1e-15)

            x = x_new
            Fx = Fx_new

            if norm(Fx) <= 1e-8:
                return x, Fx

        return x, Fx

    candidates = [np.array(x0, dtype=float)]
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            candidates.append(np.array(x0, dtype=float) + scale * np.array(dx, dtype=float))

    best_x = None
    best_norm = float('inf')

    for x_start in candidates:
        if time_left() < 1.0:
            break

        x, Fx = broyden(F, x_start)
        n = norm(Fx)

        if n < best_norm:
            best_norm = n
            best_x = x

        if best_norm <= 1e-8:
            return best_x

    return best_x if best_x is not None else np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
