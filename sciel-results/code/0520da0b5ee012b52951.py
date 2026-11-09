"""Spectral Chebyshev collocation ODE integrator for the Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] using
    a spectral Chebyshev collocation method with Newton-Raphson.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # 320s safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n_dims = len(y0)
    
    # Try different numbers of collocation points
    # Start with a moderate number, increase if needed
    best_y = None
    best_error = np.inf
    
    for n_points in [50, 100, 200, 400]:
        if time.time() > deadline:
            break
        
        # Chebyshev-Gauss-Lobatto nodes on [t0, t1]
        # n_points+1 nodes total
        N = n_points
        # Nodes: x_k = cos(pi * k / N) for k = 0, ..., N
        # Map from [-1, 1] to [t0, t1]
        x = np.cos(np.pi * np.arange(N + 1) / N)  # in [-1, 1], descending
        t_nodes = 0.5 * (t1 - t0) * x + 0.5 * (t0 + t1)  # map to [t0, t1], descending
        # Reverse to get ascending order
        t_nodes = t_nodes[::-1]
        x = x[::-1]
        
        # Derivative matrix for Chebyshev-Gauss-Lobatto nodes
        # D is the (N+1) x (N+1) matrix such that (D * y) gives y' at the nodes
        c = np.array([(-1)**k for k in range(N + 1)])
        c[0] *= 2
        c[-1] *= 2
        
        # Compute derivative matrix
        D = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                if i == j:
                    D[i, j] = -0.5 * x[i] / (1 - x[i]**2) if abs(1 - x[i]**2) > 1e-15 else 0
                else:
                    D[i, j] = c[i] / (c[j] * (x[i] - x[j]))
        
        # Scale from [-1, 1] to [t0, t1]
        D = D * (t1 - t0) / 2
        
        # Initial guess: interpolate y0 to all nodes (constant)
        y = np.tile(y0.reshape(-1, 1), (1, N + 1))  # shape (n_dims, N+1)
        
        # Newton-Raphson iterations
        max_newton = 50
        tol = 1e-14
        
        converged = False
        for newton_iter in range(max_newton):
            if time.time() > deadline:
                break
            
            # Residual: r_k = D @ y - f(t_k, y_k) for k = 1, ..., N (skip k=0 which is initial condition)
            # Actually, we enforce y(0) = y0, so we only need to satisfy the ODE at nodes 1..N
            # Let's work with the full system but fix y[:, 0] = y0
            
            # Compute D @ y
            Dy = D @ y  # shape (N+1, n_dims)
            
            # Compute f at all nodes
            f_vals = np.zeros((N + 1, n_dims))
            for k in range(N + 1):
                f_vals[k] = f(t_nodes[k], y[:, k])
            
            # Residual at nodes 1..N
            residual = (Dy[1:] - f_vals[1:]).T  # shape (n_dims * N,)
            
            # Check convergence
            res_norm = np.linalg.norm(residual)
            if res_norm < tol:
                converged = True
                break
            
            # Compute Jacobian of the residual with respect to y[:, 1..N]
            # J is a (n_dims * N) x (n_dims * N) matrix
            # For simplicity, use finite differences for the Jacobian
            # This is expensive but works for small systems
            
            n_unknowns = n_dims * N  # number of unknowns (y at nodes 1..N)
            
            # Build Jacobian using finite differences
            J = np.zeros((n_unknowns, n_unknowns))
            eps = 1e-8
            
            # Vectorize y[:, 1:] into a single vector
            y_vec = y[:, 1:].ravel()
            
            for j in range(n_unknowns):
                if time.time() > deadline:
                    break
                
                # Perturb
                y_pert = y.copy()
                idx_dim = j // N
                idx_node = j % N + 1  # node index (1-based in the full array)
                
                delta = eps * max(1.0, abs(y_vec[j]))
                y_pert[idx_dim, idx_node] += delta
                
                # Compute new residual
                Dy_pert = D @ y_pert
                f_vals_pert = np.zeros((N + 1, n_dims))
                for k in range(1, N + 1):
                    f_vals_pert[k] = f(t_nodes[k], y_pert[:, k])
                
                residual_pert = (Dy_pert[1:] - f_vals_pert[1:]).T.ravel()
                
                J[:, j] = (residual_pert - residual) / delta
            
            # Solve J @ delta_y = -residual
            try:
                delta_y = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                # Use least squares if singular
                delta_y = np.linalg.lstsq(J, -residual, rcond=None)[0]
            
            # Update y[:, 1:]
            y_vec_new = y_vec + delta_y
            y[:, 1:] = y_vec_new.reshape(n_dims, N)
            
            # Keep y[:, 0] = y0
            y[:, 0] = y0
        
        if not converged:
            continue
        
        # The solution at t1 is y[:, -1] (last node)
        y_end = y[:, -1]
        
        # Check if finite
        if not np.all(np.isfinite(y_end)):
            continue
        
        # Estimate error by comparing with a slightly different n_points
        # For now, just store this as a candidate
        if best_y is None or True:
            best_y = y_end
        
        # If we've converged well, we can break
        if res_norm < 1e-15:
            break
    
    # If no convergence, fall back to a simple method
    if best_y is None or not np.all(np.isfinite(best_y)):
        # Fallback: use scipy's solve_ivp if available
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-11, atol=1e-13, dense_output=False)
            if sol.success:
                best_y = sol.y[:, -1]
            else:
                # Last resort: RK4 with many steps
                h = (t1 - t0) / 10000
                t, y = t0, y0.copy()
                for _ in range(10000):
                    k1 = f(t, y)
                    k2 = f(t + h/2, y + h/2 * k1)
                    k3 = f(t + h/2, y + h/2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                    t += h
                best_y = y
        except Exception:
            # Absolute last resort
            h = (t1 - t0) / 10000
            t, y = t0, y0.copy()
            for _ in range(10000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            best_y = y
    
    return best_y
# EVOLVE-BLOCK-END
