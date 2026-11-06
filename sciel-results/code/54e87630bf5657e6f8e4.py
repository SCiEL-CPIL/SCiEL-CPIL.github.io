"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear ODE y' = A y on [t0, t1].
    Exploits linearity to extract A with n calls, then uses matrix exponential.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    if time.time() > deadline:
        return y0

    # Extract the system matrix A
    # Assuming f(t, y) = A y
    A = np.zeros((n, n), dtype=float)
    t_probe = t0

    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t_probe, e_j)

    if time.time() > deadline:
        # Fallback if extraction failed or timed out (unlikely)
        return y0

    # Compute the state transition matrix Phi(t1, t0) = exp(A * (t1 - t0))
    dt = t1 - t0
    Phi = expm(A * dt)

    # Calculate final state
    y_end = Phi @ y0

    return y_end
# EVOLVE-BLOCK-END
