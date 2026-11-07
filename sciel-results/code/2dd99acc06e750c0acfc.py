"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: kill at 330s, safety margin 30s
_DEADLINE = 330.0
_SAFETY_MARGIN = 30.0


def _rk45_step(f, t, y, h):
    """Single step of RK45 (Dormand-Prince). Returns (y_new, err_est)."""
    a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1, 1/2
    b21 = 1/4
    b31, b32 = 3/32, 9/32
    b41, b42, b43 = -9/32, 3/4, -3/16
    b51, b52, b53, b54 = 3/4, -9/11, 18/11, -5/11
    b61, b62, b63, b64, b65 = -21/55, 27/55, -12/11, 8/11, 5/22

    c2, c3, c4, c5, c6 = a2, a3, a4, a5, a6

    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*(b21*k1))
    k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
    k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = f(t + c6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

    # 4th order solution
    y4 = y + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
    # 5th order solution
    y5 = y + h*(5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k6)

    # Error estimate
    err = np.max(np.abs(y5 - y4)) / np.maximum(1.0, np.max(np.abs(y5)))
    return y5, err


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-11, atol=1e-13, max_steps=500000):
    """Adaptive RK45 integrator with error control."""
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 1000.0
    h_min = 1e-12
    h_max = (t1 - t0) / 10.0
    steps = 0
    start_time = time.time()

    while t < t1 - 1e-14:
        if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
            return y
        if steps >= max_steps:
            return y

        # Clamp step to not overshoot
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min

        y_new, err = _rk45_step(f, t, y, h)
        steps += 1

        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue

        # Accept or reject
        scale = atol + rtol * np.max(np.abs(y_new))
        if err <= 1.0 or h <= h_min:
            t += h
            y = y_new
            # Adjust step size
            if err > 0:
                factor = 0.9 * (1.0 / err) ** 0.2
            else:
                factor = 5.0
            h = h * min(max(factor, 0.1), 5.0)
            h = min(h, h_max)
        else:
            factor = 0.9 * (1.0 / err) ** 0.2
            h = h * max(factor, 0.1)
            h = max(h, h_min)

    return y


def _radau_implicit(f, t0, t1, y0, rtol=1e-11, atol=1e-13, max_steps=200000):
    """
    Implicit Radau IIA(3) - 3-stage 3rd order method.
    Uses Newton with finite-difference Jacobian.
    """
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 1000.0
    h_min = 1e-12
    h_max = (t1 - t0) / 10.0
    n = len(y)
    steps = 0
    start_time = time.time()

    # Radau IIA(3) coefficients
    # c1 = 0.3602730539134304, c2 = 0.7600887893767218, c3 = 1.0
    c1 = 0.3602730539134304
    c2 = 0.7600887893767218
    c3 = 1.0
    a11 = 0.1388600901425592
    a12 = 0.0
    a13 = 0.0
    a21 = 0.3532084140466746
    a22 = 0.0958924601658734
    a23 = 0.0
    a31 = 0.5515689023097173
    a32 = 0.3585633879569826
    a33 = 0.0898677097333001

    b1 = 0.5515689023097173
    b2 = 0.3585633879569826
    b3 = 0.0898677097333001

    while t < t1 - 1e-14:
        if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
            return y
        if steps >= max_steps:
            return y

        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min

        tn = t + h

        # Stage equations: Yi = y + h * sum_j a_ij f(tn, Yj)
        # We need to solve for Y1, Y2, Y3 simultaneously (fully implicit)
        # Or solve sequentially since it's lower triangular? No, a13=0, a23=0 but a31,a32 nonzero.
        # Actually it's not fully triangular. Let's solve all three stages together.
        # State vector: [Y1; Y2; Y3], each of size n. Total size 3n.

        Y = np.zeros(3 * n)
        # Initial guess
        Y[0:n] = y
        Y[n:2*n] = y
        Y[2*n:3*n] = y

        for _newton in range(15):
            # Compute F(Y) = Y - y - h * A * f(tn, Y)
            # where A is the 3x3 matrix of a_ij
            Y1 = Y[0:n]
            Y2 = Y[n:2*n]
            Y3 = Y[2*n:3*n]

            f1 = f(tn, Y1)
            f2 = f(tn, Y2)
            f3 = f(tn, Y3)

            # Residuals
            R1 = Y1 - y - h * (a11 * f1 + a12 * f2 + a13 * f3)
            R2 = Y2 - y - h * (a21 * f1 + a22 * f2 + a23 * f3)
            R3 = Y3 - y - h * (a31 * f1 + a32 * f2 + a33 * f3)

            R = np.concatenate([R1, R2, R3])
            if np.max(np.abs(R)) <= 1e-13 * (1 + np.max(np.abs(Y))):
                break

            # Jacobian of F w.r.t. Y: I - h * (d/dY)[A * f]
            # d/dY1: I - h*[a11 J1, a12 J2, a13 J3] but J_i = df/dY_i
            # For simplicity, use finite difference on the full 3n vector
            J = np.zeros((3 * n, 3 * n))
            eps = 1e-7
            for j in range(3 * n):
                e = np.zeros(3 * n)
                e[j] = eps * max(1.0, abs(Y[j]))
                Yp = Y + e
                Y1p = Yp[0:n]
                Y2p = Yp[n:2*n]
                Y3p = Yp[2*n:3*n]
                f1p = f(tn, Y1p)
                f2p = f(tn, Y2p)
                f3p = f(tn, Y3p)
                R1p = Y1p - y - h * (a11 * f1p + a12 * f2p + a13 * f3p)
                R2p = Y2p - y - h * (a21 * f1p + a22 * f2p + a23 * f3p)
                R3p = Y3p - y - h * (a31 * f1p + a32 * f2p + a33 * f3p)
                Rp = np.concatenate([R1p, R2p, R3p])
                J[:, j] = (Rp - R) / e[j]

            try:
                step = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                step = -R * 0.1

            Y = Y + step

        Y1 = Y[0:n]
        Y2 = Y[n:2*n]
        Y3 = Y[2*n:3*n]

        # New y
        y_new = y + h * (b1 * f(tn, Y1) + b2 * f(tn, Y2) + b3 * f(tn, Y3))

        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue

        # Error estimate: use embedded lower-order method
        # For Radau IIA(3), we can estimate error using a simpler method
        # Simple approach: compare with RK4 on same step
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y_rk4 = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        err = np.max(np.abs(y_new - y_rk4)) / np.maximum(1.0, np.max(np.abs(y_new)))

        t = tn
        y = y_new
        steps += 1

        # Adjust step
        if err > 0:
            factor = 0.9 * (1.0 / err) ** 0.2
        else:
            factor = 5.0
        h = h * min(max(factor, 0.1), 5.0)
        h = min(h, h_max)
        h = max(h, h_min)

    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)

    # Use adaptive RK45 with tight tolerances
    # The Van der Pol with mu=5 is moderately stiff but RK45 should handle it
    # with adaptive step size
    y = _adaptive_rk45(f, t0, t1, y0, rtol=1e-11, atol=1e-13)

    if not np.all(np.isfinite(y)):
        # Fallback to Radau
        y = _radau_implicit(f, t0, t1, y0, rtol=1e-11, atol=1e-13)

    return y
# EVOLVE-BLOCK-END
