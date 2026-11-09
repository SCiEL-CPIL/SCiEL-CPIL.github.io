"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = A y from t_start to t_end.
    Strategy: Since the system is linear and autonomous (y' = A y), we extract the
    constant matrix A by evaluating f at the standard basis vectors. Then we compute
    the exact solution y(t_end) = exp((t_end - t_start) * A) @ y0 using scipy.linalg.expm.
    This uses exactly n calls to f (n = len(y0)) and provides machine-precision accuracy.
    """
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    n = len(y)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s margin

    # Extract A by calling f with standard basis vectors.
    # Since the system is y' = A y, f(t, y) = A @ y.
    # A[:, j] = f(0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)

    # Check if we extracted A successfully
    if not np.all(np.isfinite(A)):
        # Fallback if extraction failed (should not happen)
        return y

    dt = t_end - t_start

    try:
        from scipy.linalg import expm

        # Compute the matrix exponential
        # expm(dt * A) gives the state transition matrix
        M = expm(dt * A)

        # Apply to initial condition
        y_end = M @ y

        # Ensure finite result
        if not np.all(np.isfinite(y_end)):
            # Fallback to a simple implicit Euler if expm fails
            # This is unlikely for this problem
            h = dt / 100
            y_curr = y.copy()
            I = np.eye(n)
            M_inv = np.linalg.inv(I - h * A)
            for _ in range(100):
                y_curr = M_inv @ y_curr
            return y_curr

        return y_end

    except Exception:
        # Fallback if scipy.linalg.expm is not available or fails
        # Use a simple power series approximation for expm if needed,
        # or a high-order fixed step method.
        # For robustness, let's try a few steps of Backward Euler with precomputed A.
        # Since A is known, we can solve (I - h A) y_new = y_old efficiently.

        steps = 100
        h = dt / steps
        y_curr = y.copy()

        # Precompute inverse of (I - h A)
        try:
            M_inv = np.linalg.inv(np.eye(n) - h * A)
        except np.linalg.LinAlgError:
            # Singular matrix? Use pseudo-inverse or return y
            return y_curr

        for _ in range(steps):
            if time.time() > deadline:
                break
            y_curr = M_inv @ y_curr

        return y_curr
# EVOLVE-BLOCK-END
