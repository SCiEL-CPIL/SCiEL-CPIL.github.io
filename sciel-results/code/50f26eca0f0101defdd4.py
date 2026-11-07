"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system IVP using scipy's DOP853 with reference-matching tolerances.
    Returns the state vector at t_span[1].
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin within 300s limit
    
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: if scipy fails, use a simple adaptive RK45
    y = y0.copy()
    t = t0
    h = 0.005
    rtol = 1e-10
    atol = 1e-12
    n = len(y)
    
    while t < t1 and time.time() < deadline:
        h = min(h, t1 - t)
        
        # RK45 (Dormand-Prince)
        c = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
        a = [
            [0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
        ]
        b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        b_err = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        
        k = [None] * 7
        k[0] = f(t, y)
        for i in range(1, 7):
            yi = y + h * sum(a[i][j] * k[j] for j in range(i))
            k[i] = f(t + c[i] * h, yi)
        
        y_new = y + h * sum(b[j] * k[j] for j in range(6))
        y_err = h * sum(b_err[j] * k[j] for j in range(7))
        
        err_norm = np.linalg.norm(y_err / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            t += h
            y = y_new
            if err_norm > 1e-3:
                h = 0.9 * h * (1.0 / err_norm) ** 0.2
            else:
                h = min(h * 1.5, 0.05)
        else:
            h = 0.9 * h * (1.0 / err_norm) ** 0.25
            h = max(h, 1e-8)
    
    return y
# EVOLVE-BLOCK-END
