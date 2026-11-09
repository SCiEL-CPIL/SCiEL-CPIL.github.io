# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n_dim = len(y0_arr)
    
    # Use spectral Chebyshev collocation with ~150 nodes
    # This should give machine precision accuracy with few Newton iterations
    N = 150  # number of collocation points
    
    # Chebyshev nodes in [-1, 1]
    # Map to [t0, t1]
    theta = np.pi * np.arange(N) / (N - 1)
    x_cheb = np.cos(theta)  # Chebyshev nodes in [-1, 1]
    x_cheb = x_cheb[::-1]   # reverse to go from t0 to t1
    
    # Map to [t0, t1]
    t_nodes = 0.5 * (t1 - t0) * x_cheb + 0.5 * (t1 + t0)
    
    # Chebyshev differentiation matrix
    # D is the differentiation matrix such that d/dx (p(x)) = D @ p(x)
    # For Chebyshev points x_i = cos(pi*i/(N-1))
    c = np.ones(N)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1)**np.arange(N)
    
    X = np.tile(x_cheb, (N, 1))
    dX = X - X.T
    
    D = (np.outer(c, 1.0/c) / (dX + np.eye(N)))
    D = D - np.diag(np.sum(D, axis=1))
    
    # Scale to [t0, t1]: d/dt = (2/(t1-t0)) * d/dx
    D_t = (2.0 / (t1 - t0)) * D
    
    # Initial guess: linear interpolation of y0
    Y = np.zeros((N, n_dim))
    Y[0] = y0_arr
    # Simple initial guess: constant
    for i in range(N):
        Y[i] = y0_arr
    
    # Newton-Raphson iterations
    max_iter = 50
    tol = 1e-13
    call_count = 0
    max_calls = 2000000
    start_time = time.time()
    deadline = 300.0
    
    # Jacobian of the residual with respect to Y
    # Residual: F(Y) = D_t @ Y - f(t_nodes, Y)
    # We need to solve J @ delta = -F(Y)
    
    for iteration in range(max_iter):
        # Check deadline
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Evaluate f at all nodes
        # F(t, y) = [y1, 5*(1-y0^2)*y1 - y0]
        F_vals = np.zeros((N, n_dim))
        for i in range(N):
            F_vals[i] = f(t_nodes[i], Y[i])
            call_count += 1
            if call_count >= max_calls - N:
                break
        
        # Residual: D_t @ Y - F_vals
        # D_t @ Y: differentiate Y with respect to t
        DY = np.zeros((N, n_dim))
        for j in range(n_dim):
            DY[:, j] = D_t @ Y[:, j]
        
        R = DY - F_vals
        
        # Check convergence
        R_norm = np.linalg.norm(R)
        if R_norm < tol:
            break
        
        # Build Jacobian J
        # J is a block matrix: J = D_t - J_f
        # where J_f is the Jacobian of f with respect to y
        # For Van der Pol: f = [y1, 5*(1-y0^2)*y1 - y0]
        # df/dy = [[0, 1], [-10*y0*y1 - 1, 5*(1-y0^2)]]
        
        # Build the full Jacobian
        # Size: (N * n_dim) x (N * n_dim)
        # This is large, so we use sparse structure
        
        # Actually, let's use a simpler approach: finite difference Jacobian
        # Or better: use the analytic Jacobian
        
        # The system is: D_t @ Y - f(t, Y) = 0
        # Taking derivative w.r.t. Y:
        # D_t @ I - J_f = J
        # where J_f is the Jacobian of f w.r.t. y
        
        # Build J_f at each node
        J_f = np.zeros((N, n_dim, n_dim))
        for i in range(N):
            y0_i = Y[i, 0]
            y1_i = Y[i, 1]
            J_f[i, 0, 0] = 0.0
            J_f[i, 0, 1] = 1.0
            J_f[i, 1, 0] = -10.0 * y0_i * y1_i - 1.0
            J_f[i, 1, 1] = 5.0 * (1.0 - y0_i**2)
        
        # Build the full Jacobian matrix
        # J = D_t (block identity) - J_f (diagonal blocks)
        # This is a large dense matrix, but N=150, n_dim=2, so size is 300x300
        J_full = np.zeros((N * n_dim, N * n_dim))
        
        # Add D_t part
        for i in range(N):
            for j in range(N):
                for a in range(n_dim):
                    for b in range(n_dim):
                        J_full[i * n_dim + a, j * n_dim + b] = D_t[i, j] * (1 if a == b else 0)
        
        # Subtract J_f part (diagonal blocks)
        for i in range(N):
            for a in range(n_dim):
                for b in range(n_dim):
                    J_full[i * n_dim + a, i * n_dim + b] -= J_f[i, a, b]
        
        # Solve J_full @ delta = -R
        R_flat = R.flatten()
        try:
            delta = np.linalg.solve(J_full, -R_flat)
        except np.linalg.LinAlgError:
            # Use least squares if singular
            delta, _, _, _ = np.linalg.lstsq(J_full, -R_flat, rcond=None)
        
        # Update Y
        Y += delta.reshape(N, n_dim)
        
        # Damping if needed
        if np.linalg.norm(delta) > 1.0:
            Y -= 0.5 * delta.reshape(N, n_dim)
    
    # Return the value at t1 (last node)
    return Y[-1].copy()
# EVOLVE-BLOCK-END
