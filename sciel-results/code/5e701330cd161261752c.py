"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve the Robertson stiff ODE using a high-order implicit Runge-Kutta method
    with analytic Jacobian, leveraging scipy's Radau solver for efficiency.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: ensure we finish well before the 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 300.0  # 300s safety margin
    
    # Robertson problem: y' = f(t, y)
    # y[0] = A, y[1] = B, y[2] = C
    # A' = -0.04*A + 1e4*B*C
    # B' = 0.04*A - 1e4*B*C - 3e7*B^2
    # C' = 3e7*B^2
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        A, B, C = y
        return np.array([
            [-0.04, 1e4*C, 1e4*B],
            [0.04, -1e4*C - 6e7*B, -1e4*B],
            [0.0, 6e7*B, 0.0]
        ])
    
    # Use scipy's Radau solver with analytic Jacobian
    try:
        from scipy.integrate import solve_ivp
        
        # Try Radau first (implicit, good for stiff problems)
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='Radau',
            jac=jac,
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='BDF',
            jac=jac,
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # Last resort: LSODA
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception:
        pass
    
    # If all scipy methods fail, fall back to a simple implicit method
    # Backward Euler with Newton iteration and finite-difference Jacobian
    def _backward_euler(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        n = len(y)
        
        for _ in range(steps):
            z = y.copy()
            for _newton in range(10):
                if time.time() > deadline - 5.0:
                    break
                fz = f(t + h, z)
                J = np.empty((n, n))
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = 1e-8 * max(1.0, abs(z[j]))
                    J[:, j] = (f(t + h, z + e) - fz) / e[j]
                
                A = np.eye(n) - h * J
                b = -(z - y - h * fz)
                
                try:
                    step = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(A, b, rcond=None)[0]
                
                z = z + step
                if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                    break
            
            y, t = z, t + h
        
        return y
    
    return _backward_euler(f, t0, t1, y0, 5000)
# EVOLVE-BLOCK-END
