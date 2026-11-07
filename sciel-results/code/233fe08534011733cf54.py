import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=np.float64)
    y = y0.copy()

    # Adaptive step RK45-like scheme with error estimation.
    # We use a fixed sequence of steps that refines until error is small.
    # To minimize calls, we use large initial steps and only refine if needed.

    # First, do a single large step to see if we're in trouble
    # But we need accuracy to ~1e-14 relative, which is very tight.
    # For Lorenz over [0,2], the solution is chaotic but bounded.
    # We'll use an adaptive RK method with tight tolerances.

    # Implement a high-order adaptive integrator using RKF78 (Runge-Kutta-Fehlberg 7(8))
    # Coefficients for RKF78 (from Fehlberg 1968)
    # a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65,
    # a71, a72, a73, a74, a75, a76, a81, a82, a83, a84, a85, a86, a87,
    # a91, a92, a93, a94, a95, a96, a97, a98
    a21 = 1/9
    a31 = 1/24
    a32 = 1/24
    a41 = 1/6
    a42 = 1/3
    a43 = -1/6
    a51 = 0
    a52 = 5/72
    a53 = 1/6
    a54 = 5/24
    a61 = 1/135
    a62 = 0
    a63 = 1/10
    a64 = 3/5
    a65 = -49/90
    a71 = 5/144
    a72 = 0
    a73 = 35/168
    a74 = 1/8
    a75 = 3/16
    a76 = 1/12
    a81 = 1/12
    a82 = 0
    a83 = 0
    a84 = 1/8
    a85 = 5/12
    a86 = 1/4
    a87 = 5/12
    a91 = 5179/57600
    a92 = 0
    a93 = 7571/16695
    a94 = 393/640
    a95 = 92097/339200
    a96 = 187/2100
    a97 = 1/40
    a98 = 512/525

    # b7 (7th order)
    b7_1 = 35/384
    b7_2 = 0
    b7_3 = 0
    b7_4 = 500/1113
    b7_5 = 125/192
    b7_6 = -2187/6784
    b7_7 = 11/84
    b7_8 = 0

    # b8 (8th order)
    b8_1 = 5179/57600
    b8_2 = 0
    b8_3 = 7571/16695
    b8_4 = 393/640
    b8_5 = 92097/339200
    b8_6 = 187/2100
    b8_7 = 1/40
    b8_8 = 512/525

    # c values (time increments)
    c2 = a21
    c3 = a31 + a32
    c4 = a41 + a42 + a43
    c5 = a51 + a52 + a53 + a54
    c6 = a61 + a62 + a63 + a64 + a65
    c7 = a71 + a72 + a73 + a74 + a75 + a76
    c8 = a81 + a82 + a83 + a84 + a85 + a86 + a87
    c9 = a91 + a92 + a93 + a94 + a95 + a96 + a97 + a98

    rtol = 1e-14
    atol = 1e-16

    t = t0
    h = (t1 - t0) / 100.0  # initial step estimate
    max_h = (t1 - t0) / 5.0
    min_h = (t1 - t0) * 1e-6
    max_steps = 50000

    steps = 0
    while t < t1 and steps < max_steps:
        # Clamp h to remaining time
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h

        # Compute RK stages
        k1 = f(t, y)

        y2 = y + h * a21 * k1
        k2 = f(t + c2 * h, y2)

        y3 = y + h * (a31 * k1 + a32 * k2)
        k3 = f(t + c3 * h, y3)

        y4 = y + h * (a41 * k1 + a42 * k2 + a43 * k3)
        k4 = f(t + c4 * h, y4)

        y5 = y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k5 = f(t + c5 * h, y5)

        y6 = y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        k6 = f(t + c6 * h, y6)

        y7 = y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        k7 = f(t + c7 * h, y7)

        y8 = y + h * (a81 * k1 + a82 * k2 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)
        k8 = f(t + c8 * h, y8)

        y9 = y + h * (a91 * k1 + a92 * k2 + a93 * k3 + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8)
        k9 = f(t + c9 * h, y9)

        # 7th order solution
        y7_sol = y + h * (b7_1 * k1 + b7_4 * k4 + b7_5 * k5 + b7_6 * k6 + b7_7 * k7)
        # 8th order solution
        y8_sol = y + h * (b8_1 * k1 + b8_3 * k3 + b8_4 * k4 + b8_5 * k5 + b8_6 * k6 + b8_7 * k7 + b8_8 * k8)

        # Error estimate: difference between 7th and 8th order
        err = np.max(np.abs(y8_sol - y7_sol))
        scale = np.max(np.abs(y8_sol))
        err_rel = err / max(scale, 1e-30)

        if err_rel <= rtol:
            # Accept step
            y = y8_sol
            t += h
            steps += 1
            # Increase step size
            if err_rel > 1e-20:
                h_new = h * min(5.0, max(0.2, 0.9 * (rtol / err_rel) ** 0.2))
            else:
                h_new = h * 5.0
            h = min(h_new, max_h)
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * (rtol / err_rel) ** 0.25)
            if h < min_h:
                h = min_h
            steps += 1

    return y
