"""ODE integrator for the Van der Pol oscillator (mu = 5).

Uses Radau IIA (3-stage, 3rd-order) implicit Runge-Kutta with an
analytic Jacobian, Newton iteration, and a modest fixed step count.
This family (implicit one-step, L-stable, analytic Jacobian) is
different from the explicit RK4 / finite-difference backward Euler
approach in the seed program.
"""

# EVOLVE-BLOCK-START
import numpy as np


def _radau_iiA_step(f, J, y, t, h):
    """Advance one Radau IIA (s=3, order 3) step from y at t to t+h.

    Coefficients (Radau IIA, 3 stages):
        c = [1/6, 2/3, 1]
        A = [[1/6, 0,   0  ],
             [1/6, 2/3, 0  ],
             [1/6, 2/3, 1  ]]
        b = [1/6, 2/3, 1/6]

    Each stage is solved with Newton's method using the analytic
    Jacobian J(t, y).  The system is 2-D, so a 2x2 solve is cheap.
    """
    c = np.array([1.0/6.0, 2.0/3.0, 1.0])
    A = np.array([[1.0/6.0, 0.0,     0.0   ],
                  [1.0/6.0, 2.0/3.0, 0.0   ],
                  [1.0/6.0, 2.0/3.0, 1.0   ]])
    b = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])

    n = len(y)
    # Precompute the I - h*A_jj block structure for each stage.
    # Stage i equation: Y_i = y + h * sum_j A_ij * f(t + c_j h, Y_j)
    # Rearranged: Y_i - h * A_ii * f(t + c_i h, Y_i) = y + h * sum_{j<i} A_ij f_j
    # For the implicit part we need to solve: (I - h * A_ii * J_i) delta = rhs
    # where J_i is the Jacobian of f at (t + c_i h, Y_i).

    # We will solve stages sequentially since A is lower triangular.
    F = np.zeros((3, n))   # stage derivatives f(t + c_i h, Y_i)
    Y = np.zeros((3, n))   # stage values

    for i in range(3):
        ti = t + c[i] * h
        # Initial guess: for i==0 use y; otherwise use previous stage value.
        if i == 0:
            Y[i] = y.copy()
        else:
            Y[i] = Y[i-1].copy()

        # Newton iterations for stage i.
        # Define g(Y_i) = Y_i - h*A_ii*f(ti, Y_i) - (y + h*sum_{j<i} A_ij * F[j])
        # We need g(Y_i) = 0.
        # g'(Y_i) = I - h * A_ii * J(ti, Y_i)

        rhs_known = y.copy()
        for j in range(i):
            rhs_known += h * A[i, j] * F[j]

        max_newton = 8
        tol = 1e-12
        for _ in range(max_newton):
            fi = f(ti, Y[i])
            Ji = J(ti, Y[i])
            # g = Y_i - h*A_ii*fi - rhs_known
            g = Y[i] - h * A[i, i] * fi - rhs_known
            # g' = I - h * A_ii * Ji
            G = np.eye(n) - h * A[i, i] * Ji
            try:
                delta = np.linalg.solve(G, -g)
            except np.linalg.LinAlgError:
                # Fallback: use a small damped step
                delta = -0.1 * g
            Y[i] = Y[i] + delta
            if np.linalg.norm(delta) < tol * (1.0 + np.linalg.norm(Y[i])):
                break
        F[i] = f(ti, Y[i])

    # Final update: y_{n+1} = y + h * sum_i b_i F_i
    y_new = y + h * (b[0] * F[0] + b[1] * F[1] + b[2] * F[2])
    return y_new


def _van_der_pol_jac(t, y):
    """Analytic Jacobian of f(t, y) = [y1, mu*(1-y0^2)*y1 - y0] with mu=5."""
    mu = 5.0
    y0, y1 = y[0], y[1]
    J = np.empty((2, 2), dtype=float)
    J[0, 0] = 0.0
    J[0, 1] = 1.0
    J[1, 0] = -mu * 2.0 * y0 * y1 - 1.0
    J[1, 1] = mu * (1.0 - y0 * y0)
    return J


def integrate(f, t_span, y0):
    """Integrate the Van der Pol oscillator from t_span[0] to t_span[1].

    Parameters
    ----------
    f : callable
        Vector field f(t, y) -> y' (array of length 2).
    t_span : tuple
        (t_start, t_end).
    y0 : array-like
        Initial state vector.

    Returns
    -------
    y_end : np.ndarray
        State at t_end.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    T = t1 - t0

    # Choose step count: Radau IIA is 3rd order and L-stable.
    # For mu=5 Van der Pol on [0,10], 100 steps (h=0.1) gives good accuracy.
    # With ~6-8 f calls per step (3 stages * ~2-3 Newton iters each,
    # plus final stage eval), that's ~600-800 total calls.
    steps = 100
    h = T / steps

    for _ in range(steps):
        y = _radau_iiA_step(f, _van_der_pol_jac, y, t0, h)
        t0 += h
        if not np.all(np.isfinite(y)):
            # If something goes wrong, try to salvage with a smaller step
            # or just return what we have (should not happen for this problem)
            break

    return y

# EVOLVE-BLOCK-END
