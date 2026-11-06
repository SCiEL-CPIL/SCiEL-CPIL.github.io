import numpy as np
import scipy.linalg as sla
import scipy.integrate as si

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Step 1: Estimate A
    A = np.zeros((N, N), dtype=float)
    for j in range(N):
        e_j = np.zeros(N, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        
    # Step 2: Check linearity
    # Pick a random vector
    v = np.random.rand(N)
    fv = f(0.0, v)
    Av = A @ v
    # Check if f(v) approx A v
    # Use relative error
    err = np.linalg.norm(fv - Av) / (np.linalg.norm(Av) + 1e-12)
    
    if err < 1e-8:
        # Assume linear
        dt = t1 - t0
        M = sla.expm(A * dt)
        y_end = M @ y0
        return y_end
    else:
        # Fallback to implicit solver
        # Radau is good for stiff problems
        sol = si.solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-10, atol=1e-12)
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = si.solve_ivp(f, (t0, t1), y0, method='BDF', rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
            # Last resort: RK4 with small steps (like seed but maybe more steps if time allows?)
            # But if we are here, we are in trouble.
            # Just return the best we have? Or raise?
            # The task says "return the final state".
            # Let's try a dense RK4 if implicit fails.
            steps = 10000
            h = (t1 - t0) / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
