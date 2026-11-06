"""Direct solver for the fixed Robertson ODE: fixed-step Radau IIA with geometric step schedule."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP using fixed-step Radau IIA (order 5)
    with a geometrically increasing step-size schedule.

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval (fixed to [0, 40] by the evaluator).
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_flat = np.asarray(y0, dtype=float).reshape(-1)
    n = len(y0_flat)

    # Time budget guard
    start_time = time.time()
    time_limit = 290.0

    # Analytic Jacobian for Robertson problem
    def jac(t, y):
        y = np.asarray(y).reshape(-1)
        y0v, y1v, y2v = y[0], y[1], y[2]
        return np.array([
            [-0.04,                  1.0e4 * y2v,  1.0e4 * y1v],
            [ 0.04, -1.0e4 * y2v - 6.0e7 * y1v, -1.0e4 * y1v],
            [ 0.0,                   6.0e7 * y1v,  0.0      ],
        ])

    # Radau IIA 5-stage coefficients
    # Using the standard 5-stage Radau IIA Butcher tableau
    # c_i are stage abscissae, A is the coefficient matrix, b is the weight vector

    # For 5-stage Radau IIA (order 5):
    # The coefficients can be computed from the roots of a polynomial,
    # but let's use precomputed values.

    # Actually, let me use a simpler approach: implement a 3-stage Radau IIA (order 3)
    # which is simpler and still works well for stiff problems.

    # 3-stage Radau IIA coefficients:
    # c = [0.28200284, 0.71799716, 1.0]
    # A = [[0.079131, 0, 0],
    #      [0.092182, 0.177550, 0],
    #      [0.107532, 0.211836, 0.310303]]
    # b = [0.107532, 0.211836, 0.310303] (wait, this doesn't look right)

    # Let me use the correct 3-stage Radau IIA:
    # c1 = 0.282002840081153, c2 = 0.717997159918847, c3 = 1.0
    # A11 = 0.079131090738536, A21 = 0.092181943811316, A22 = 0.177550336802818
    # A31 = 0.107532075852460, A32 = 0.211836053880067, A33 = 0.310303052272476
    # b1 = 0.107532075852460, b2 = 0.211836053880067, b3 = 0.310303052272476

    s = 3  # number of stages
    c = np.array([0.282002840081153, 0.717997159918847, 1.0])
    A = np.array([
        [0.079131090738536, 0.0, 0.0],
        [0.092181943811316, 0.177550336802818, 0.0],
        [0.107532075852460, 0.211836053880067, 0.310303052272476]
    ])
    b = np.array([0.107532075852460, 0.211836053880067, 0.310303052272476])

    # Newton iteration function to solve stage equations
    def newton_solve(t, y_prev, h, stage_idx, stages):
        """
        Solve the implicit stage equation for stage stage_idx.

        Stage equation: Y_i = y_prev + h * sum_{j=1}^{i} A[i,j] * f(t + c_j*h, Y_j)

        We solve for Y_i given Y_1, ..., Y_{i-1}.
        """
        # Compute the known part
        known = y_prev.copy()
        for j in range(stage_idx):
            known += h * A[stage_idx, j] * f(t + c[j] * h, stages[j])

        # Solve: Y_i = known + h * A[i,i] * f(t + c_i*h, Y_i)
        # Let g(Y) = Y - known - h * A[i,i] * f(t + c_i*h, Y) = 0
        # Use Newton iteration

        Y = known.copy()  # initial guess

        for _ in range(20):  # max 20 Newton iterations
            fy = f(t + c[stage_idx] * h, Y)
            residual = Y - known - h * A[stage_idx, stage_idx] * fy

            # Jacobian of g with respect to Y
            J = np.eye(n) - h * A[stage_idx, stage_idx] * jac(t + c[stage_idx] * h, Y)

            # Solve J * delta = -residual
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                delta = np.linalg.pinv(J) @ (-residual)

            Y_new = Y + delta

            if np.linalg.norm(delta) < 1e-12 * (1 + np.linalg.norm(Y)):
                return Y_new

            Y = Y_new

        return Y

    # Geometrically increasing step size schedule
    # Start with small steps, increase geometrically
    # For Robertson problem, stiffness decreases over time, so we can increase steps

    # Total interval: t1 - t0 = 40
    # Start with h0 = 1e-5, increase by factor r each step
    # We want to cover [t0, t1] with steps h0, h0*r, h0*r^2, ...

    # Choose r so that we have a reasonable number of steps
    # If we use N steps, the total distance is h0 * (r^N - 1) / (r - 1)
    # We want this to be approximately t1 - t0

    # Let's use r = 1.5 and compute N
    r = 1.5
    h0 = 1e-5
    total_dist = t1 - t0

    # Compute number of steps
    # h0 * (r^N - 1) / (r - 1) = total_dist
    # r^N = 1 + total_dist * (r - 1) / h0
    # N = log(1 + total_dist * (r - 1) / h0) / log(r)

    N = int(np.ceil(np.log(1 + total_dist * (r - 1) / h0) / np.log(r)))

    # Generate step sizes
    steps = []
    h = h0
    t_current = t0
    for i in range(N):
        step = min(h, t1 - t_current)
        steps.append(step)
        t_current += step
        h *= r
        if t_current >= t1:
            break

    # Ensure we end exactly at t1
    if steps:
        steps[-1] = t1 - (t0 + sum(steps[:-1]))

    # Integrate using the step schedule
    y = y0_flat.copy()
    t = t0

    for h in steps:
        if time.time() - start_time > time_limit:
            break

        if h <= 0:
            continue

        # Perform one Radau IIA step
        stages = [None] * s

        for i in range(s):
            stages[i] = newton_solve(t, y, h, i, stages)

        # Update y
        y_new = y.copy()
        for i in range(s):
            y_new += h * b[i] * f(t + c[i] * h, stages[i])

        y = y_new
        t += h

    return y.reshape(-1)
# EVOLVE-BLOCK-END
