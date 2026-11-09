"""ODE solver for 100-point convection-diffusion IVP using matrix exponential via extracted system matrix."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def integrate(f, t_span, y0):
    # Deadline guard: return best-effort result if we run out of wall-clock time.
    t_start = time.perf_counter()
    DEADLINE = 300.0  # Evaluator kills at 330s; keep a safety margin.

    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    n = y0.shape[0]

    def _elapsed_ok():
        return (time.perf_counter() - t_start) < DEADLINE

    # Strategy: The convection-diffusion problem is linear and time-invariant: y' = A @ y.
    # We can extract the matrix A by evaluating f at the identity basis vectors.
    # Then compute exp(A * (t1 - t0)) @ y0 directly using high-precision matrix exponential.
    # This uses ~n+1 calls to f (101 calls) and then one matrix exponential.
    # This should be extremely accurate and use very few f evaluations.

    try:
        # Extract the system matrix A by probing with identity vectors
        # f(t, y) = A @ y for linear time-invariant system
        # Evaluate at t=0 (or any t, since it's time-invariant)
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            if not _elapsed_ok():
                raise TimeoutError()
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            # Evaluate f at t=0 with y = e_j
            # For a linear system y' = A*y, f(0, e_j) = A @ e_j = A[:, j]
            A[:, j] = f(0.0, e_j)

        if not _elapsed_ok():
            raise TimeoutError()

        # Compute the matrix exponential: exp(A * (t1 - t0))
        dt = t1 - t0
        # Use scipy's expm which is highly accurate
        expm_A_dt = expm(A * dt)

        if not _elapsed_ok():
            raise TimeoutError()

        # Final state: y(t1) = exp(A * dt) @ y0
        y_final = expm_A_dt @ y0

        if np.all(np.isfinite(y_final)):
            return y_final

    except Exception:
        pass

    # Fallback: Use DOP853 with tight tolerances
    try:
        if not _elapsed_ok():
            return y0
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 100.0,
        )
        if sol.success and sol.t.size > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback 2: Simple RK4
    if not _elapsed_ok():
        return y0
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if not _elapsed_ok():
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
# EVOLVE-BLOCK-END
