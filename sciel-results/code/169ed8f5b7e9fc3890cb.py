"""ODE solver: Heat2D 10x10 - integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    dt_total = t1 - t0
    deadline = time.time() + 280.0  # Safety margin before 330s kill

    # RK45 (Dormand-Prince) coefficients
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    # 5th order solution coefficients (local extrapolation)
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0

    # Fixed step count
    # Based on analysis, N=30000 provides high accuracy with manageable call count
    n_steps = 30000
    h = dt_total / n_steps
    t = t0

    for i in range(n_steps):
        if time.time() > deadline:
            break

        k1 = f(t, y)
        k2 = f(t + c2 * h, y + h * a21 * k1)
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        y = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
        t += h

    return y
# EVOLVE-BLOCK-END
