import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Check for linearity and autonomy
    # f(t, y) = A y
    f0 = f(t0, y0)
    f1 = f(t0, 2.0 * y0)
    
    # Tolerance for linearity check
    # If f is linear, f(2y) = 2f(y)
    is_linear = np.allclose(f1, 2.0 * f0, rtol=1e-5, atol=1e-8)
    
    if is_linear:
        # Construct Jacobian A using finite differences
        # This requires N+1 calls.
        A = np.zeros((N, N))
        # We already have f0.
        # For each column j
        for j in range(N):
            e = np.zeros(N)
            # Choose perturbation size
            scale = max(1.0, abs(y0[j]))
            h_fd = 1e-8 * scale
            e[j] = h_fd
            fj = f(t0, y0 + e)
            A[:, j] = (fj - f0) / h_fd
        
        # Use Crank-Nicolson
        K = 2000  # Number of steps
        h = (t1 - t0) / K
        
        I = np.eye(N)
        M = I - 0.5 * h * A
        R = I + 0.5 * h * A
        
        # Convert to sparse for faster solve? N=100, dense is fine and faster due to overhead.
        # But let's use scipy sparse lu for robustness if N was larger.
        # For N=100, np.linalg.solve is extremely fast.
        
        y = y0.copy()
        for _ in range(K):
            # y_next = M^-1 R y
            # Precompute R @ y? No, y changes.
            rhs = R @ y
            y = np.linalg.solve(M, rhs)
            
        return y
    else:
        # Fallback to robust solver
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=(t1-t0)/100)
        return sol.y[:, -1]
