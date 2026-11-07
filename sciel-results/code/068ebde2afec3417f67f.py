"""Root finder for 3 fixed 2D systems using Newton with periodic Jacobian recompute and merit line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses Newton's method with periodic finite-difference Jacobian recomputation
    and a merit-function line search for robust global convergence.
    Falls back to Broyden updates between Newton steps to save F-calls.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    tol = 1e-8
    start_time = time.time()
    time_limit = 138.0  # Safety margin for 150s kill

    def time_left():
        return time_limit - (time.time() - start_time)

    def norm(v):
        return np.linalg.norm(v)

    # Initial evaluation
    best_x = x.copy()
    Fx = F(best_x)
    best_norm = norm(Fx)

    if best_norm <= tol:
        return best_x

    # Finite-difference Jacobian (reuses Fx, costs 2 extra F calls)
    def fd_jacobian(x_pt, Fx_pt):
        h = 1e-7 * max(1.0, norm(x_pt))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x_pt + e) - Fx_pt) / h
        return J

    # Merit function: m(x) = 0.5 * ||F(x)||^2
    def merit(Fx_pt):
        return 0.5 * Fx_pt @ Fx_pt

    # Solve J dx = -Fx with fallback regularization
    def solve_jacobian(J, rhs):
        try:
            return np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            try:
                return np.linalg.solve(J + 1e-12 * np.eye(2), rhs)
            except np.linalg.LinAlgError:
                return None

    # Main Newton solver with periodic Jacobian recompute and merit line search
    def newton_solve(x_start, max_iters=60):
        nonlocal best_x, best_norm

        x_local = x_start.copy()
        Fx_local = F(x_local)
        norm_local = norm(Fx_local)
        m_local = merit(Fx_local)

        if norm_local < best_norm:
            best_norm = norm_local
            best_x = x_local.copy()
        if norm_local <= tol:
            return x_local

        # Initial Jacobian via FD (2 F calls)
        J = fd_jacobian(x_local, Fx_local)
        jac_age = 0  # iterations since last Jacobian recompute
        max_jac_age = 5  # recompute Jacobian every 5 iterations

        for i in range(max_iters):
            if time_left() < 3.0:
                break
            if norm_local <= tol:
                return x_local

            # Periodically recompute Jacobian for accuracy
            if jac_age >= max_jac_age:
                J = fd_jacobian(x_local, Fx_local)
                jac_age = 0

            # Newton direction
            dx = solve_jacobian(J, -Fx_local)
            if dx is None:
                # Fallback: recompute Jacobian and try again
                J = fd_jacobian(x_local, Fx_local)
                jac_age = 0
                dx = solve_jacobian(J, -Fx_local)
                if dx is None:
                    break

            # Merit line search (Armijo)
            alpha = 1.0
            c1 = 1e-4  # Armijo constant
            grad_m = Fx_local @ J  # gradient of merit = J^T F
            # Actually for merit m = 0.5||F||^2, grad = J^T F, and directional derivative = grad^T dx
            dir_deriv = (J.T @ Fx_local) @ dx

            accepted = False
            for ls in range(20):
                if time_left() < 2.0:
                    break
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = norm(F_try)
                m_try = merit(F_try)

                # Armijo condition: m(x + alpha*dx) <= m(x) + c1 * alpha * dir_deriv
                if m_try <= m_local + c1 * alpha * dir_deriv:
                    # Accept step
                    s = alpha * dx
                    y = F_try - Fx_local

                    # Broyden rank-1 update for next iteration (saves FD calls)
                    Js = J @ s
                    denom = s @ s
                    if denom > 1e-16:
                        J += np.outer(y - Js, s) / denom

                    x_local = x_try
                    Fx_local = F_try
                    norm_local = norm_try
                    m_local = m_try
                    jac_age += 1
                    accepted = True

                    if norm_local < best_norm:
                        best_norm = norm_local
                        best_x = x_local.copy()
                    if norm_local <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                # If line search failed, try a smaller step or break
                if time_left() < 2.0:
                    break
                # Try a very small step in the negative gradient direction
                grad = J.T @ Fx_local
                gnorm = norm(grad)
                if gnorm > 1e-16:
                    x_try = x_local - 0.01 * grad / gnorm
                    F_try = F(x_try)
                    norm_try = norm(F_try)
                    if norm_try < norm_local:
                        x_local = x_try
                        Fx_local = F_try
                        norm_local = norm_try
                        m_local = merit(F_try)
                        if norm_local < best_norm:
                            best_norm = norm_local
                            best_x = x_local.copy()
                    else:
                        break
                else:
                    break

        return x_local

    # Broyden solver (lighter, for restarts when Newton struggles)
    def broyden_solve(x_start, max_iters=30):
        nonlocal best_x, best_norm

        x_local = x_start.copy()
        Fx_local = F(x_local)
        norm_local = norm(Fx_local)

        if norm_local < best_norm:
            best_norm = norm_local
            best_x = x_local.copy()
        if norm_local <= tol:
            return x_local

        J = fd_jacobian(x_local, Fx_local)

        for i in range(max_iters):
            if time_left() < 3.0:
                break
            if norm_local <= tol:
                return x_local

            dx = solve_jacobian(J, -Fx_local)
            if dx is None:
                break

            # Simple backtracking
            alpha = 1.0
            accepted = False
            for ls in range(10):
                if time_left() < 2.0:
                    break
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = norm(F_try)
                if norm_try < norm_local:
                    s = alpha * dx
                    y = F_try - Fx_local
                    Js = J @ s
                    denom = s @ s
                    if denom > 1e-16:
                        J += np.outer(y - Js, s) / denom
                    x_local = x_try
                    Fx_local = F_try
                    norm_local = norm_try
                    accepted = True
                    if norm_local < best_norm:
                        best_norm = norm_local
                        best_x = x_local.copy()
                    if norm_local <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                break

        return x_local

    # Main loop: try Newton first, then Broyden, with strategic restarts
    current_x = x.copy()
    n_restarts = 0
    max_restarts = 20

    while n_restarts < max_restarts and time_left() > 8.0:
        if best_norm <= tol:
            return best_x

        # Primary: Newton with merit line search
        result = newton_solve(current_x)
        if best_norm <= tol:
            return best_x

        n_restarts += 1

        # If Newton got close, try Broyden from the best point for polish
        if best_norm < 1e-5 and time_left() > 10.0:
            result = broyden_solve(best_x, max_iters=20)
            if best_norm <= tol:
                return best_x

        # Choose next starting point based on restart count
        if n_restarts == 1:
            # Small perturbation around x0
            current_x = x + np.random.normal(0, 0.5, size=2)
        elif n_restarts == 2:
            # Around best_x, moderate radius
            radius = min(3.0, 0.5 * n_restarts)
            d = np.random.randn(2)
            d /= norm(d)
            current_x = best_x + radius * d
        elif n_restarts == 3:
            # Wider exploration around x0
            current_x = x + np.random.uniform(-3, 3, size=2)
        elif n_restarts % 3 == 0:
            # Periodic return to neighborhood of x0
            current_x = x + np.random.normal(0, 1.0, size=2)
        elif n_restarts % 3 == 1:
            # Around best_x with growing radius
            radius = min(8.0, 0.4 * n_restarts)
            d = np.random.randn(2)
            d /= norm(d)
            current_x = best_x + radius * d
        else:
            # Broader search
            current_x = x + np.random.uniform(-5, 5, size=2)

    # Final polish: if we're close, do a few Newton iterations with fresh Jacobians
    if best_norm < 1e-4 and time_left() > 5.0:
        x_local = best_x.copy()
        Fx_local = F(x_local)
        norm_local = norm(Fx_local)
        for _ in range(20):
            if time_left() < 2.0 or norm_local <= tol:
                break
            J = fd_jacobian(x_local, Fx_local)
            dx = solve_jacobian(J, -Fx_local)
            if dx is None:
                break
            x_try = x_local + dx
            F_try = F(x_try)
            norm_try = norm(F_try)
            if norm_try < norm_local:
                x_local = x_try
                Fx_local = F_try
                norm_local = norm_try
                if norm_local < best_norm:
                    best_norm = norm_local
                    best_x = x_local.copy()
            else:
                # Try half step
                x_try = x_local + 0.5 * dx
                F_try = F(x_try)
                norm_try = norm(F_try)
                if norm_try < norm_local:
                    x_local = x_try
                    Fx_local = F_try
                    norm_local = norm_try
                    if norm_local < best_norm:
                        best_norm = norm_local
                        best_x = x_local.copy()
                else:
                    break

    return best_x
# EVOLVE-BLOCK-END
