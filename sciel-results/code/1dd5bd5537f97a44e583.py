import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0

    # Reconstruct A via basis vectors
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
    
    if time.time() > deadline:
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]

    # Verify linearity / time-invariance with a couple of probes
    # Use a deterministic non-trivial vector
    y_probe = np.linspace(1.0, 2.0, n)
    t_probe = 0.3 * dt
    y_lin = A @ y_probe
    y_act = f(t_probe, y_probe)
    err = np.linalg.norm(y_act - y_lin)
    if err > 1e-13 * (1.0 + np.linalg.norm(y_lin)):
        # Not linear / time-invariant; fall back to high-accuracy ODE solver
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-13, atol=1e-15)
        return sol.y[:, -1]

    # Use matrix exponential for exact linear autonomous solution
    M = A * dt
    try:
        expM = scipy.linalg.expm(M)
        y_end = expM @ y0
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    # Fallback: high-accuracy adaptive solver
    sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-13, atol=1e-15)
    return sol.y[:, -1]
