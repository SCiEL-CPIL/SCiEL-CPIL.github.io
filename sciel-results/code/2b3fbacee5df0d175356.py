"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk4_adaptive(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=100000, deadline=None):
    """Adaptive RK45 (Cash-Karp) with step-size control."""
    # Cash-Karp coefficients
    a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1.0
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    # 4th order weights
    c4 = [2825/27648, 0, 18575/46656, 13525/46656, 4275/3382, 0]
    # 5th order weights
    c5 = [37/378, 0, 250/621, 125/594, 0, 512/1771]
    sigmas = [1/4, 1/6, 1/6, 1/24, 1/24, 1/12]

    y = np.array(y0, dtype=float)
    t = t0
    n = len(y)
    h = (t1 - t0) / 1000.0
    if h <= 0:
        return y

    # Initial step size
    y0_norm = np.linalg.norm(y)
    scale = atol + rtol * y0_norm
    if y0_norm > 0:
        f0 = f(t, y)
        f0_norm = np.linalg.norm(f0)
        if f0_norm > 0:
            h = 0.01 * y0_norm / f0_norm
            h = min(h, t1 - t0)
            h = max(h, 1e-12)

    steps = 0
    while t < t1:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        if steps >= max_steps:
            break

        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t

        # Cash-Karp stages
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*b21*k1)
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

        y4 = y + h * (c4[0]*k1 + c4[2]*k3 + c4[3]*k4 + c4[4]*k5)
        y5 = y + h * (c5[0]*k1 + c5[2]*k3 + c5[3]*k4 + c5[5]*k6)

        err = np.linalg.norm(y5 - y4)
        scale_vec = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_norm = np.linalg.norm((y5 - y4) / scale_vec)

        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            steps += 1

            # Adjust step size
            if err_norm > 1e-10:
                h *= min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                h *= 5.0
        else:
            # Reject step, reduce
            h *= max(0.1, 0.9 * err_norm ** (-0.25))
            if h < 1e-14:
                h = 1e-14

    return y


def _rk4_fixed(f, t0, t1, y0, steps):
    """Fixed-step RK4 as fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin

    # Try adaptive RK45 first
    result = _rk4_adaptive(f, t0, t1, y0, rtol=1e-13, atol=1e-15,
                           max_steps=200000, deadline=deadline)
    if result is not None and np.all(np.isfinite(result)):
        return result

    # Fallback: fixed-step RK4 with moderate steps
    result = _rk4_fixed(f, t0, t1, y0, 5000)
    if result is not None and np.all(np.isfinite(result)):
        return result

    # Last resort: even more steps
    result = _rk4_fixed(f, t0, t1, y0, 10000)
    if result is not None:
        return result

    # Absolute last resort
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
