"""Root finder for 3 fixed 2D systems using homotopy continuation with Broyden polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses homotopy continuation to track the solution path, then Broyden's method to polish.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    start_time = time.time()
    time_limit = 130.0  # Safety margin for 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - elapsed()

    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))

    # Broyden's method for polishing
    def broyden_solve(x_start, max_iters=30):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        # Initial Jacobian via finite differences (3 F calls)
        h = 1e-5
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = F(x_local + e)
            J[:, j] = (Fx_pert - Fx) / h

        for i in range(max_iters):
            if time_left() < 2.0:
                break

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-8 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Backtracking line search
            alpha = 1.0
            accepted = False
            for _ in range(6):
                if time_left() < 1.0:
                    break
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx:
                    s = alpha * dx
                    y = F_try - Fx
                    Js = J @ s
                    denom = s @ s
                    if denom > 1e-14:
                        J += np.outer(y - Js, s) / denom
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    accepted = True
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                    if norm_Fx <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                break

        return x_local

    # Homotopy continuation: track root of H(x, t) = (1-t)*F(x) + t*(x - x0)
    # At t=0, H(x,0) = F(x), root is the target
    # At t=1, H(x,1) = x - x0, root is x0
    # We go from t=1 to t=0, starting at x0
    def homotopy_continuation(x_start, max_steps=200):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        t = 1.0
        dt = 0.05  # step size in t

        for step in range(max_steps):
            if time_left() < 3.0:
                break
            if t <= 0:
                break

            # Solve H(x, t) = 0 for x given current t
            # H(x, t) = (1-t)*F(x) + t*(x - x_start)
            # Use a few Newton-like iterations with Broyden
            def H(x_curr, t_curr):
                return (1 - t_curr) * F(x_curr) + t_curr * (x_curr - x_start)

            # Inner Broyden solve for fixed t
            x_inner = x_local.copy()
            Hx = H(x_inner, t)
            norm_Hx = np.linalg.norm(Hx)
            
            # Compute Jacobian of H: dH/dx = (1-t)*J_F + t*I
            # Approximate J_F with finite differences
            h_fd = 1e-5
            J_H = np.eye(2) * t
            Fx_inner = F(x_inner)
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_fd
                Fx_pert = F(x_inner + e)
                J_H[:, j] += (1 - t) * (Fx_pert - Fx_inner) / h_fd

            # Newton iterations for inner solve
            for inner_iter in range(10):
                if time_left() < 2.0:
                    break
                try:
                    dx = np.linalg.solve(J_H, -Hx)
                except np.linalg.LinAlgError:
                    break

                # Damped step
                alpha = 1.0
                accepted = False
                for _ in range(4):
                    if time_left() < 1.0:
                        break
                    x_try = x_inner + alpha * dx
                    H_try = H(x_try, t)
                    norm_try = np.linalg.norm(H_try)
                    if norm_try < norm_Hx:
                        x_inner = x_try
                        Hx = H_try
                        norm_Hx = norm_try
                        accepted = True
                        break
                    else:
                        alpha *= 0.5

                if not accepted:
                    break

                if norm_Hx < 1e-10:
                    break

            x_local = x_inner
            t -= dt

            # Track best
            Fx_check = F(x_local)
            norm_check = np.linalg.norm(Fx_check)
            if norm_check < best_norm:
                best_norm = norm_check
                best_x = x_local.copy()
            if best_norm <= tol:
                return x_local

        return x_local

    # Strategy:
    # 1. Try homotopy continuation from x0
    # 2. If not converged, try Broyden from the homotopy result
    # 3. If still not converged, try random restarts with Broyden

    # Phase 1: Homotopy continuation
    result = homotopy_continuation(x, max_steps=150)
    if best_norm <= tol:
        return best_x

    # Phase 2: Polish with Broyden from homotopy result
    result = broyden_solve(result, max_iters=20)
    if best_norm <= tol:
        return best_x

    # Phase 3: Try Broyden from x0 directly
    result = broyden_solve(x, max_iters=30)
    if best_norm <= tol:
        return best_x

    # Phase 4: Random restarts
    np.random.seed(42)
    for i in range(20):
        if time_left() < 5.0:
            break
        if best_norm <= tol:
            return best_x

        # Choose random starting point
        if i % 3 == 0:
            x_try = x + np.random.normal(0, 1.0, size=2)
        elif i % 3 == 1:
            radius = min(10.0, 0.5 * (i + 1))
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            x_try = best_x + radius * dir_vec
        else:
            x_try = x + np.random.uniform(-5, 5, size=2)

        result = broyden_solve(x_try, max_iters=25)
        if best_norm <= tol:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
