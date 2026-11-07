import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0
    max_calls = 490  # Safety margin below 500

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    def compute_jacobian_fd(x_pt, Fx_pt):
        """Compute Jacobian via forward finite differences (2 calls)."""
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt + e2)
            J = np.array([(Fx1 - Fx_pt) / h, (Fx2 - Fx_pt) / h]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Initial Jacobian estimate
    J = compute_jacobian_fd(x, Fx)
    if J is None:
        J = np.eye(2)

    # Main Loop: Broyden's Method with Line Search
    max_iter = 50
    stagnation_count = 0

    for i in range(max_iter):
        if not time_ok() or call_count > max_calls:
            break

        if norm_sq <= tol_sq:
            return x

        # Update best
        if norm_sq < best_norm_sq:
            best_norm_sq = norm_sq
            best_x = x.copy()

        # Solve for Newton step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                dx = np.linalg.solve(J + 1e-6 * np.eye(2), -Fx)
            except:
                # If still failing, restart with perturbation
                x = x + 0.1 * np.random.randn(2)
                try:
                    Fx = F_count(x)
                except:
                    return best_x
                if not np.all(np.isfinite(Fx)):
                    return best_x
                norm_sq = float(np.dot(Fx, Fx))
                J = compute_jacobian_fd(x, Fx)
                if J is None:
                    J = np.eye(2)
                continue

        # Line Search
        # We want to find lambda such that F(x + lambda*dx) is smaller
        # Simple backtracking
        lambda_val = 1.0
        x_try = x + lambda_val * dx
        Fx_try = None
        norm_sq_try = None
        
        # Try full step first
        try:
            Fx_try = F_count(x_try)
            if np.all(np.isfinite(Fx_try)):
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
            else:
                Fx_try = None
        except:
            Fx_try = None

        if Fx_try is None or norm_sq_try > norm_sq:
            # Backtrack
            for _ in range(5):
                lambda_val *= 0.5
                x_try = x + lambda_val * dx
                try:
                    Fx_try = F_count(x_try)
                    if not np.all(np.isfinite(Fx_try)):
                        continue
                    norm_sq_try = float(np.dot(Fx_try, Fx_try))
                    if norm_sq_try < norm_sq:
                        break
                except:
                    continue
        
        if Fx_try is None:
            # Failed to find a good step, try restart
            x = x + 0.1 * np.random.randn(2)
            try:
                Fx = F_count(x)
            except:
                return best_x
            if not np.all(np.isfinite(Fx)):
                return best_x
            norm_sq = float(np.dot(Fx, Fx))
            J = compute_jacobian_fd(x, Fx)
            if J is None:
                J = np.eye(2)
            continue

        # Accept step
        x_actual_step = lambda_val * dx
        Fx_old = Fx  # Save old F for Broyden update
        x = x_try
        Fx = Fx_try
        norm_sq = norm_sq_try

        # Update Jacobian (Broyden Rank-1)
        dF = Fx - Fx_old
        Jdx = J @ x_actual_step
        residual = dF - Jdx
        dx_norm_sq = float(np.dot(x_actual_step, x_actual_step))
        
        if dx_norm_sq > 1e-30:
            J = J + np.outer(residual, x_actual_step) / dx_norm_sq

        # Check for stagnation
        if norm_sq > best_norm_sq * 0.99:
            stagnation_count += 1
        else:
            stagnation_count = 0
            
        if stagnation_count > 5:
            # Restart with perturbation
            x = x + 0.1 * np.random.randn(2)
            try:
                Fx = F_count(x)
            except:
                return best_x
            if not np.all(np.isfinite(Fx)):
                return best_x
            norm_sq = float(np.dot(Fx, Fx))
            J = compute_jacobian_fd(x, Fx)
            if J is None:
                J = np.eye(2)
            stagnation_count = 0

    return best_x
