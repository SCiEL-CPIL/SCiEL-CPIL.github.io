"""Root finder for 3 fixed 2D systems using Powell's secant method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses Powell's secant method (2D) which requires exactly 1 F-call per iteration
    after initial setup, plus a Newton polish pass for final convergence.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    tol = 1e-8
    start_time = time.time()
    time_limit = 135.0  # Safety margin for 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - elapsed()

    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))

    if best_norm <= tol:
        return best_x

    # Newton polish: high-accuracy final convergence using FD Jacobian
    def polish(x_start, max_iters=20):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        h = 1e-6 * max(1.0, np.linalg.norm(x_local))
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            if norm_Fx <= tol:
                return x_local

            # Compute Jacobian via central differences (4 F calls)
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = F(x_local + e)
                Fx_m = F(x_local - e)
                J[:, j] = (Fx_p - Fx_m) / (2 * h)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-12 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Damped Newton step
            alpha = 1.0
            improved = False
            for _ in range(8):
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx:
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                    improved = True
                    break
                alpha *= 0.5

            if not improved:
                break

        return x_local

    # Powell's 2D secant method
    # For n=2, we maintain A = [[a, b]] where A represents the 1x2 matrix
    # such that the step is computed as: dx = A @ Fx
    # The update uses the secant equation.
    def powell_secant(x_start, max_iters=60):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        # Initialize A (1x2 matrix) using FD Jacobian inverse
        h = 1e-5 * max(1.0, np.linalg.norm(x_local))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = F(x_local + e)
            J[:, j] = (Fx_pert - Fx) / h

        try:
            A = np.linalg.inv(J).T  # A is 1x2: A = (J^{-1})^T
        except np.linalg.LinAlgError:
            # Fallback: use identity-like initialization
            A = np.eye(2)

        for i in range(max_iters):
            if time_left() < 2.0:
                break
            if norm_Fx <= tol:
                return x_local

            # Compute step: dx = A @ Fx (A is 1x2, Fx is 2x1)
            dx = A @ Fx

            # Check if step is reasonable
            step_norm = np.linalg.norm(dx)
            if step_norm < 1e-14:
                break

            # Damped step with backtracking
            alpha = 1.0
            accepted = False
            for _ in range(15):
                if time_left() < 1.0:
                    break
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)

                if norm_try < norm_Fx:
                    # Secant update for A (1x2 matrix)
                    s = alpha * dx  # 2x1
                    y = F_try - Fx  # 2x1

                    # A_new = A - (A @ y) * (A @ s)^{-1} * (s^T @ A^T)
                    # More carefully: A_new = A - (A @ y) @ inv(A @ s) @ (s^T @ A^T)
                    # A @ y is scalar, A @ s is scalar, s^T @ A^T is 1x2
                    Ay = float(A @ y)  # scalar
                    As = float(A @ s)  # scalar

                    if abs(As) > 1e-14:
                        # A_new = A - (Ay / As) * (s^T @ A^T)
                        # s^T @ A^T is 1x2: (A @ s)^T ... wait
                        # Let me re-derive. A is 1x2.
                        # A_new = A - (A y)(s^T A^T) / (A s)
                        # A y is scalar (1x1), s^T A^T is 1x2, A s is scalar (1x1)
                        A = A - (Ay / As) * (s @ A)  # s @ A is 1x2
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                    accepted = True
                    if norm_Fx <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                break

        return x_local

    # Main loop with restarts
    current_x = x.copy()
    n_restarts = 0
    max_restarts = 30

    while n_restarts < max_restarts and time_left() > 5.0:
        if best_norm <= tol:
            return best_x

        # Try Powell's secant method from current_x
        result = powell_secant(current_x)
        if best_norm <= tol:
            return best_x

        n_restarts += 1

        # Choose new starting point
        if n_restarts % 5 == 0:
            # Periodically try points near x0
            current_x = x + np.random.normal(0, 1.0, size=2)
        elif n_restarts % 5 == 1:
            # Try around best_x with growing radius
            radius = min(10.0, 0.5 * n_restarts)
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            current_x = best_x + radius * dir_vec
        elif n_restarts % 5 == 2:
            # Try in a wider box around x0
            current_x = x + np.random.uniform(-5, 5, size=2)
        elif n_restarts % 5 == 3:
            # Try around best_x with moderate radius
            radius = min(5.0, 0.3 * n_restarts)
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            current_x = best_x + radius * dir_vec
        else:
            # Try a random point in a reasonable range
            current_x = x + np.random.uniform(-10, 10, size=2)

    # Final polish pass if we got close but didn't quite converge
    if best_norm < 1e-4 and time_left() > 5.0:
        polish(best_x, max_iters=30)

    return best_x
# EVOLVE-BLOCK-END
