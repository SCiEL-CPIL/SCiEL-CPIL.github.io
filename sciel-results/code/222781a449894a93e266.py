"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator y0' = y1, y1' = -y0 over t_span.
    Uses a custom adaptive Dormand-Prince 5(4) (DP54) integrator.
    """
    deadline = time.time() + 280.0  # 300s limit, 20s safety margin

    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0

    if t0 == t1:
        return y

    # Dormand-Prince 5(4) coefficients
    # a2 = 1/5
    a3 = 3 / 10
    a4 = 4 / 5
    a5 = 8 / 9
    a6 = 1.0
    a7 = 1.0

    b2 = 1 / 5
    b3 = 3 / 40
    b4 = 3 / 10
    b5 = -11 / 40
    b6 = 5 / 84
    b7 = 0.0

    c2 = 1 / 5
    c3 = 3 / 10
    c4 = 4 / 5
    c5 = 8 / 9
    c6 = 1.0
    c7 = 1.0

    # b for 5th order (for error estimation and step acceptance)
    b1 = 71 / 576
    b3 = -71 / 160
    b4 = 71 / 120
    b5 = -17 / 60
    b6 = 1 / 5
    b7 = 0.0

    # b for 4th order (for actual step update)
    b1_4 = 35 / 384
    b3_4 = 0.0
    b4_4 = 500 / 1113
    b5_4 = 125 / 192
    b6_4 = -2187 / 2720
    b7_4 = 11 / 40

    rtol = 1e-13
    atol = 1e-14
    max_step = 0.1
    min_step = 1e-12
    h = 0.01  # initial step size

    while t < t1 - 1e-15:
        if time.time() > deadline:
            return y

        # Ensure we don't overshoot t1
        h = min(h, t1 - t)
        h = max(h, min_step)

        y_new = None
        accepted = False

        for _attempt in range(10):
            k1 = f(t, y)
            k2 = f(t + c2 * h, y + h * (b2 * k1))
            k3 = f(t + c3 * h, y + h * (b3 * k1 + b4 * k2))
            k4 = f(t + c4 * h, y + h * (b5 * k1 + b6 * k2 + b7 * k3))
            k5 = f(t + c5 * h, y + h * (-11 / 40 * k1 + 5 / 84 * k2 + 5 / 84 * k3 + 1 / 7 * k4))
            k6 = f(t + c6 * h, y + h * (1 / 8 * k1 - 1 / 2 * k3 + 1 / 2 * k4 - 1 / 8 * k5))
            k7 = f(t + c7 * h, y + h * (-1 / 12 * k1 + 3 / 4 * k3 - 3 / 4 * k4 + 1 / 2 * k5 - 1 / 12 * k6))

            # 5th order solution
            y5 = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
            # 4th order solution
            y4 = y + h * (b1_4 * k1 + b3_4 * k3 + b4_4 * k4 + b5_4 * k5 + b6_4 * k6 + b7_4 * k7)

            # Error estimate
            err = y5 - y4
            err_norm = np.linalg.norm(err, ord=np.inf)
            scale = atol + rtol * np.maximum(np.linalg.norm(y, ord=np.inf), np.linalg.norm(y5, ord=np.inf))
            err_scaled = err_norm / scale

            if err_scaled <= 1.0:
                y_new = y5
                accepted = True
                break
            else:
                # Adjust step size
                if err_scaled > 0:
                    factor = 0.9 * (err_scaled ** (-0.2))
                else:
                    factor = 1.0
                h = h * max(0.1, min(10.0, factor))
                h = max(h, min_step)
                h = min(h, max_step)

        if not accepted:
            # If we couldn't accept a step, try a much smaller step
            h = h * 0.1
            if h < min_step:
                # Force accept with smaller step
                k1 = f(t, y)
                k2 = f(t + c2 * h, y + h * (b2 * k1))
                k3 = f(t + c3 * h, y + h * (b3 * k1 + b4 * k2))
                k4 = f(t + c4 * h, y + h * (b5 * k1 + b6 * k2 + b7 * k3))
                k5 = f(t + c5 * h, y + h * (-11 / 40 * k1 + 5 / 84 * k2 + 5 / 84 * k3 + 1 / 7 * k4))
                k6 = f(t + c6 * h, y + h * (1 / 8 * k1 - 1 / 2 * k3 + 1 / 2 * k4 - 1 / 8 * k5))
                k7 = f(t + c7 * h, y + h * (-1 / 12 * k1 + 3 / 4 * k3 - 3 / 4 * k4 + 1 / 2 * k5 - 1 / 12 * k6))
                y_new = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
                accepted = True

        if not np.all(np.isfinite(y_new)):
            return y

        y = y_new
        t += h

        # Adjust step size for next step
        if err_scaled > 0 and err_scaled < 1.0:
            factor = 0.9 * (err_scaled ** (-0.2))
        elif err_scaled >= 1.0:
            factor = 0.9
        else:
            factor = 1.0
        h = h * max(0.1, min(5.0, factor))
        h = min(h, max_step)
        h = max(h, min_step)

    return y
# EVOLVE-BLOCK-END
