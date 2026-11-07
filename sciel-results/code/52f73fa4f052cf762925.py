"""Root finder for 3 fixed 2D systems: damped Broyden with adaptive line-search and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start = time.time()
    deadline = start + 105.0  # safety margin under 120s limit

    x = np.array(x0, dtype=float)
    if not (np.all(np.isfinite(x))):
        x = np.zeros(2)

    # first evaluation
    Fx = np.asarray(F(x), dtype=float).ravel()
    if not np.all(np.isfinite(Fx)):
        return x
    best_x = x.copy()
    best_norm = float(np.linalg.norm(Fx))
    if best_norm <= 1e-8:
        return best_x

    # initial Jacobian via forward differences
    h = 1e-6
    J = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        Fp = np.asarray(F(x + e), dtype=float).ravel()
        J[:, j] = (Fp - Fx) / h
    if not np.all(np.isfinite(J)):
        J = np.eye(2)

    def time_ok():
        return time.time() < deadline

    # Broyden rank-1 update
    def update_broyden(J, s, y):
        denom = float(np.dot(y, s))
        if abs(denom) < 1e-14:
            return J
        return J + np.outer((y - J @ s), s) / denom

    max_iter = 400
    for it in range(max_iter):
        if not time_ok():
            break
        # solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx * 1e-3  # pseudo-step
        if not np.all(np.isfinite(dx)):
            dx = -Fx * 1e-3

        # adaptive damping / line search
        lam = 1.0
        accepted = False
        for _ in range(12):
            xn = x + lam * dx
            Fxn = np.asarray(F(xn), dtype=float).ravel()
            if np.all(np.isfinite(Fxn)):
                nrm = float(np.linalg.norm(Fxn))
                if nrm < best_norm:
                    best_x = xn.copy()
                    best_norm = nrm
                if nrm < best_norm * 0.999 or nrm <= 1e-8:
                    # accept
                    y = Fxn - Fx
                    s = lam * dx
                    J = update_broyden(J, s, y)
                    x = xn
                    Fx = Fxn
                    accepted = True
                    break
            lam *= 0.5
        if accepted:
            if best_norm <= 1e-8:
                return best_x
            continue
        # no acceptable damped step: re-estimate Jacobian and try a small direct step
        h2 = 1e-6
        Jnew = np.empty((2, 2))
        for j in range(2):
            e = np.zeros(2); e[j] = h2
            Fp = np.asarray(F(x + e), dtype=float).ravel()
            Jnew[:, j] = (Fp - Fx) / h2
        if np.all(np.isfinite(Jnew)):
            J = Jnew
        # tiny step along -Fx
        xn = x - 1e-3 * Fx
        Fxn = np.asarray(F(xn), dtype=float).ravel()
        if np.all(np.isfinite(Fxn)):
            nrm = float(np.linalg.norm(Fxn))
            if nrm < best_norm:
                best_x = xn.copy()
                best_norm = nrm
                y = Fxn - Fx
                s = xn - x
                J = update_broyden(J, s, y)
                x = xn
                Fx = Fxn
        else:
            break

    return best_x
# EVOLVE-BLOCK-END
