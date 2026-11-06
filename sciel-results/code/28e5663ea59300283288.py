import numpy as np
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # 1. Probe for linearity and homogeneity
    # Evaluate at 0
    y_zero = f(t0, np.zeros(N))
    
    # Check if f is roughly linear
    # Build A from basis vectors
    A = np.zeros((N, N))
    for j in range(N):
        e_j = np.zeros(N)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Check homogeneity
    b = y_zero
    # Adjust A for inhomogeneous part if needed?
    # If f(t,y) = A_lin y + b, then f(t, e_j) = A_lin e_j + b.
    # So A_lin[:, j] = A[:, j] - b.
    A_lin = A - b[:, np.newaxis]
    
    # Verify linearity with a random vector
    y_test = np.random.rand(N)
    y_pred = A_lin @ y_test + b
    y_act = f(t0, y_test)
    
    # Tolerance for linearity check
    # Scale relative to magnitude
    norm_act = np.linalg.norm(y_act)
    norm_diff = np.linalg.norm(y_act - y_pred)
    
    if norm_diff < 1e-8 * (1 + norm_act):
        # System is linear
        dt = t1 - t0
        try:
            # Compute exponential
            # For homogeneous (b=0): y_end = expm(A_lin * dt) @ y0
            # For inhomogeneous: y_end = expm(A_lin * dt) @ y0 + (expm(A_lin * dt) - I) @ inv(A_lin) @ b
            # To avoid explicit inverse, solve A_lin x = b
            M = scipy.linalg.expm(A_lin * dt)
            y_end = M @ y0
            if np.linalg.norm(b) > 1e-10:
                # Solve A_lin x = b
                try:
                    x = scipy.linalg.solve(A_lin, b)
                    y_end += (M - np.eye(N)) @ x
                except np.linalg.LinAlgError:
                    # Fallback if singular
                    # Use pseudo-inverse
                    x = scipy.linalg.pinv(A_lin) @ b
                    y_end += (M - np.eye(N)) @ x
            return y_end
        except Exception:
            pass # Fallback to solver
            
    # Fallback: Stiff ODE Solver
    # Radau is good for stiff problems
    sol = scipy.integrate.solve_ivp(
        f, t_span, y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12,
        jac='2-point' # Let it estimate Jacobian, or we can pass None
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        # Last resort: RK4 with small steps?
        # Or BDF
        sol2 = scipy.integrate.solve_ivp(
            f, t_span, y0, 
            method='BDF', 
            rtol=1e-8, 
            atol=1e-10
        )
        return sol2.y[:, -1]
