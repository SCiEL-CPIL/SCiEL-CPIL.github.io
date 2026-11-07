"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 290.0  # safety margin before 330s limit

    t0, t1 = t_span
    y = np.array(y0, dtype=float)

    # Dormand-Prince 5(4) with fixed step size
    # For Lorenz over [0,2], 2000 steps gives h=0.001
    steps = 2000
    h = (t1 - t0) / steps

    # DOP54 Butcher tableau
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 11/54]
    # a matrix (only lower triangle needed):
    # a[1] = 1/5
    # a[2] = 3/10, a[3] = 0 (wait, let me use the standard form)
    
    # Standard DOP54:
    # k1 = f(t, y)
    # k2 = f(t+h/5, y + h/5*k1)
    # k3 = f(t+3h/10, y + h*(3/40*k1 + 9/40*k2))
    # k4 = f(t+4h/5, y + h*(44/45*k1 - 56/15*k2 + 32/9*k3))
    # k5 = f(t+8h/9, y + h*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
    # k6 = f(t+h, y + h*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 - 49/k15*k4 + 1725/556*k5))  -- this is wrong
    # Let me use the correct coefficients
    
    # Correct DOP54:
    # a1 = 0
    # a2 = 1/5
    # a31 = 3/40, a32 = 9/40
    # a41 = 44/45, a42 = -56/15, a43 = 32/9
    # a51 = 19372/6561, a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
    # a61 = 9017/3168, a62 = -355/33, a63 = 46732/5247, a64 = -49/176, a65 = 1725/556
    # a71 = 34061/330240, a72 = 0, a73 = 35277/51200, a74 = 189/512, a75 = 11/40, a76 = -8404/5120
    
    # b (5th order): 35/384, 0, 500/1113, 125/192, 2187/6784, 11/84, 25/192
    # b* (4th order): 55/576, 0, 700/1188, 350/1188, 50/1188, 0, 1/40

    t = t0
    for _ in range(steps):
        if time.time() > deadline - 5.0:
            break

        k1 = f(t, y)
        k1 = np.asarray(k1, dtype=float)

        y2 = y + h / 5.0 * k1
        k2 = f(t + h / 5.0, y2)
        k2 = np.asarray(k2, dtype=float)

        y3 = y + h * (3.0/40.0 * k1 + 9.0/40.0 * k2)
        k3 = f(t + 3.0*h/10.0, y3)
        k3 = np.asarray(k3, dtype=float)

        y4 = y + h * (44.0/45.0 * k1 - 56.0/15.0 * k2 + 32.0/9.0 * k3)
        k4 = f(t + 4.0*h/5.0, y4)
        k4 = np.asarray(k4, dtype=float)

        y5 = y + h * (19372.0/6561.0 * k1 - 25360.0/2187.0 * k2 + 64448.0/6561.0 * k3 - 212.0/729.0 * k4)
        k5 = f(t + 8.0*h/9.0, y5)
        k5 = np.asarray(k5, dtype=float)

        y6 = y + h * (9017.0/3168.0 * k1 - 355.0/33.0 * k2 + 46732.0/5247.0 * k3 - 49.0/176.0 * k4 + 1725.0/556.0 * k5)
        k6 = f(t + h, y6)
        k6 = np.asarray(k6, dtype=float)

        y7 = y + h * (34061.0/330240.0 * k1 + 35277.0/51200.0 * k3 + 189.0/512.0 * k4 + 11.0/40.0 * k5 - 8404.0/5120.0 * k6)
        k7 = f(t + 11.0*h/54.0, y7)
        k7 = np.asarray(k7, dtype=float)

        # 5th order update
        y = y + h * (35.0/384.0 * k1 + 500.0/1113.0 * k3 + 125.0/192.0 * k4 + 2187.0/6784.0 * k5 + 11.0/84.0 * k6 + 25.0/192.0 * k7)
        t += h

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break

    return y
# EVOLVE-BLOCK-END
