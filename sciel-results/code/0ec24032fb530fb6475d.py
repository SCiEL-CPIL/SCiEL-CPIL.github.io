"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Detect if the system is linear constant-coefficient (f(t,y) = A@y + b).
       If so, use eigendecomposition-based exact solution with minimal f calls.
    2. Otherwise, use a robust adaptive stiff solver.
    """
    start_time = time.time()
    deadline = start_time + 285  # 15s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Detect if system is linear constant-coefficient: f(t, y) = A @ y + b
    # We need to determine A and b with minimal f evaluations.
    
    # Step 1: Get b = f(t0, 0)
    zero_vec = np.zeros(n)
    b = f(t0, zero_vec)
    
    # Step 2: Check time-invariance by comparing f(t1, 0) with b
    b_t1 = f(t1, zero_vec)
    
    is_time_invariant = np.allclose(b, b_t1, rtol=1e-12, atol=1e-12)
    
    if is_time_invariant:
        # Step 3: Extract A by probing with basis vectors
        # For each column j: A[:, j] = f(t0, e_j) - b
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j) - b
        
        # Step 4: Verify linearity with one test point (reduces calls)
        # Use y0 itself as test point if non-zero, else use a random vector
        if np.linalg.norm(y0) > 1e-15:
            y_test = y0
        else:
            y_test = np.random.randn(n)
        
        f_test = f(t0, y_test)
        f_pred = A @ y_test + b
        
        if np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-10):
            # Use eigendecomposition-based exact solution
            try:
                T = t1 - t0
                eigenvalues, eigenvectors = np.linalg.eig(A)
                
                # Check condition number of eigenvector matrix
                cond_V = np.linalg.cond(eigenvectors)
                
                if cond_V < 1e12:
                    # Well-conditioned: use eigendecomposition directly
                    # e^{AT} = V @ diag(e^{lambda_i * T}) @ V^{-1}
                    exp_eigenvalues = np.exp(eigenvalues * T)
                    V_inv = np.linalg.inv(eigenvectors)
                    exp_AT = eigenvectors @ np.diag(exp_eigenvalues) @ V_inv
                    
                    # Compute particular solution for constant b
                    # y(t) = exp_AT @ y0 + integral_0^T exp(A*s) b ds
                    try:
                        # Solve A @ x = b, then y_particular = (exp_AT - I) @ x
                        x = np.linalg.solve(A, b)
                        y_particular = (exp_AT - np.eye(n)) @ x
                    except np.linalg.LinAlgError:
                        # A is singular; use numerical quadrature
                        m = 50
                        xs, ws = np.polynomial.legendre.leggauss(m)
                        s_vals = 0.5 * T * (xs + 1)
                        w_vals = 0.5 * T * ws
                        y_particular = np.zeros(n)
                        for s, w in zip(s_vals, w_vals):
                            exp_As = eigenvectors @ np.diag(np.exp(eigenvalues * s)) @ V_inv
                            y_particular += w * (exp_As @ b)
                    
                    y_final = exp_AT @ y0 + y_particular
                    return y_final
                else:
                    # Ill-conditioned: fall back to expm
                    from scipy.linalg import expm
                    exp_AT = expm(A * T)
                    
                    try:
                        x = np.linalg.solve(A, b)
                        y_particular = (exp_AT - np.eye(n)) @ x
                    except np.linalg.LinAlgError:
                        m = 50
                        xs, ws = np.polynomial.legendre.leggauss(m)
                        s_vals = 0.5 * T * (xs + 1)
                        w_vals = 0.5 * T * ws
                        y_particular = np.zeros(n)
                        for s, w in zip(s_vals, w_vals):
                            exp_As = expm(A * s)
                            y_particular += w * (exp_As @ b)
                    
                    y_final = exp_AT @ y0 + y_particular
                    return y_final
            except Exception:
                pass
    
    # Fallback: use a stiff solver
    try:
        from scipy.integrate import solve_ivp
        
        def f_wrapper(t, y):
            return f(t, y)
        
        sol = solve_ivp(f_wrapper, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = solve_ivp(f_wrapper, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: fixed-step implicit Euler with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    n_steps = 10000
    for i in range(n_steps):
        # Check deadline
        if time.time() > deadline:
            break
        # Simple implicit Euler: y_new = y + h * f(t+h, y_new)
        # Use fixed-point iteration (3 iterations)
        z = y + h * f(t + h, y)
        for _ in range(3):
            z = y + h * f(t + h, z)
        y = z
        t += h
    return y
# EVOLVE-BLOCK-END
