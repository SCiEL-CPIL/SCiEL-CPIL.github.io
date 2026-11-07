import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

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

    def compute_jacobian_fd(x_pt):
        """Compute Jacobian via central finite differences (4 calls)."""
        h = 1e-5
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt - e1)
            Fx3 = F_count(x_pt + e2)
            Fx4 = F_count(x_pt - e2)
            J = np.array([(Fx1 - Fx2) / (2*h), (Fx3 - Fx4) / (2*h)]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Main Loop: Broyden's Method with Adaptive Step Control
    # We allow multiple "restarts" if the method stagnates.
    max_restarts = 5
    max_iter_per_restart = 20
    
    for restart in range(max_restarts):
        if not time_ok() or call_count > 450:
            break
        
        # If this is not the first restart, we might try a perturbed starting point
        # or just rely on the current x which is the best so far.
        # For simplicity and robustness, we restart Broyden from the current best x.
        
        # Compute initial Jacobian for this restart
        J = compute_jacobian_fd(x)
        if J is None:
            J = np.eye(2)
            
        # Ensure J is invertible
        try:
            np.linalg.inv(J)
        except:
            J = J + 1e-6 * np.eye(2)

        for _ in range(max_iter_per_restart):
            if not time_ok() or call_count > 480:
                break
            
            if norm_sq <= tol_sq:
                return x

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                # If singular, add regularization
                J = J + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except:
                    break
            
            # Check if step is reasonable
            dx_norm = np.linalg.norm(dx)
            if dx_norm < 1e-12:
                # Stagnation detected
                break

            # Backtracking Line Search
            alpha = 1.0
            c1 = 1e-4 # Armijo condition constant
            rho = 0.5  # Reduction factor
            
            Fx_curr = Fx
            norm_sq_curr = norm_sq
            grad_F = Fx # Approximate gradient of 0.5*||F||^2 is J^T F, but we use F for direction
            
            # For 2D, we can just check if the new norm is smaller
            # Armijo condition: f(x + alpha*dx) <= f(x) + c1 * alpha * grad^T * dx
            # Here f(x) = 0.5 * ||F(x)||^2. grad^T dx is approximated by F^T dx (since d(0.5||F||^2) = F^T dF = F^T J dx)
            # However, a simpler and robust check for root finding is just monotonic decrease in ||F||^2.
            
            accepted = False
            for ls_iter in range(10):
                x_try = x + alpha * dx
                try:
                    Fx_try = F_count(x_try)
                except:
                    break
                if not np.all(np.isfinite(Fx_try)):
                    break
                
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                
                # Simple monotonic decrease check
                if norm_sq_try < norm_sq_curr:
                    # Accept step
                    # Update Broyden
                    s = alpha * dx
                    y = Fx_try - Fx_curr
                    s_norm_sq = float(np.dot(s, s))
                    if s_norm_sq > 1e-30:
                        J = J + np.outer(y - J @ s, s) / s_norm_sq
                    
                    x = x_try
                    Fx = Fx_try
                    norm_sq = norm_sq_try
                    accepted = True
                    
                    if norm_sq < best_norm_sq:
                        best_norm_sq = norm_sq
                        best_x = x.copy()
                    break
                else:
                    alpha *= rho
            
            if not accepted:
                # Line search failed
                break

        # If we exited the inner loop without converging, we might restart
        # The restart logic is handled by the outer loop

    return best_x
