"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Use a 6th-order explicit Runge-Kutta method with fixed step
    n_steps = 1000
    h = (t1 - t0) / n_steps
    t = t0
    
    # 7-stage 6th-order RK method coefficients
    c = [0, 1/6, 5/12, 1/2, 5/12, 1/6, 1]
    A = [
        [0, 0, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 0, 0],
        [0, 5/12, 0, 0, 0, 0, 0],
        [0, 0, 1/2, 0, 0, 0, 0],
        [0, 0, 5/12, 0, 0, 0, 0],
        [0, 1/6, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 1/6, 0]
    ]
    b = [1/6, 0, 0, 0, 0, 1/6, 0]
    
    for _ in range(n_steps):
        k = [None] * 7
        k[0] = f(t, y)
        for i in range(1, 7):
            y_stage = y + h * sum(A[i][j] * k[j] for j in range(i))
            k[i] = f(t + c[i] * h, y_stage)
        y = y + h * sum(b[i] * k[i] for i in range(7))
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
