"""ODE solver for 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Strategy:
    1. Use scipy.integrate.solve_ivp with 'BDF' or 'Radau' for stiff problems.
    2. If scipy fails, fall back to a custom implicit method.
    3. Track time to ensure we finish within 300s.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 10s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try scipy solve_ivp first - it's optimized for stiff problems
    try:
        from scipy.integrate import solve_ivp
        
        # Use Radau method (good for stiff problems) with tight tolerances
        # rtol and atol need to be small for high accuracy
        # Score = -log10(rel_err) - 0.5*log10(n_calls)
        # We want high accuracy (low rel_err) and few calls
        
        # Start with moderate tolerances, then refine if time permits
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='Radau',
            rtol=1e-12, atol=1e-14,
            max_step=0.01
        )
        
        if sol.success and sol.t[-1] >= t1:
            elapsed = time.time() - start_time
            # If we have time, try to improve accuracy
            if elapsed < deadline - 50.0:
                sol2 = solve_ivp(
                    f, [t0, t1], y0,
                    method='Radau',
                    rtol=1e-14, atol=1e-16,
                    max_step=0.005
                )
                if sol2.success and sol2.t[-1] >= t1:
                    return sol2.y[:, -1]
            return sol.y[:, -1]
        
        # If Radau failed, try BDF
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='BDF',
            rtol=1e-12, atol=1e-14,
            max_step=0.01
        )
        
        if sol.success and sol.t[-1] >= t1:
            elapsed = time.time() - start_time
            if elapsed < deadline - 50.0:
                sol2 = solve_ivp(
                    f, [t0, t1], y0,
                    method='BDF',
                    rtol=1e-14, atol=1e-16,
                    max_step=0.005
                )
                if sol2.success and sol2.t[-1] >= t1:
                    return sol2.y[:, -1]
            return sol.y[:, -1]
            
    except Exception:
        pass
    
    # Fallback: custom implicit midpoint / backward Euler with Newton
    # This is slower but should work if scipy fails
    elapsed = time.time() - start_time
    if elapsed > deadline - 30.0:
        # Not enough time for fallback, return best effort
        # Try a simple RK4 with many steps as last resort
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    
    # Implicit backward Euler with Newton
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    eye_n = np.eye(n)
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        z = y.copy()
        for _newton in range(10):
            fz = f(t + h, z)
            # Finite difference Jacobian
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = 1e-8 * max(1.0, abs(z[j]))
                e[j] = scale
                J[:, j] = (f(t + h, z + e) - fz) / scale
            # Solve (I - h*J) delta = -(z - y - h*fz)
            A = eye_n - h * J
            b = -(z - y - h * fz)
            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, b, rcond=None)[0]
            z = z + delta
            if np.linalg.norm(delta) <= 1e-12 * (1 + np.linalg.norm(z)):
                break
        y = z
        t += h
    
    return y

# EVOLVE-BLOCK-END
