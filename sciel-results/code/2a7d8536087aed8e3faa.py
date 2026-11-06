"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0.
Uses robust extraction of A and high-precision matrix exponential."""
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    deadline = time.time() + 290  # safety margin before 330s kill
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # For a linear system y' = A @ y, f(t, y) = A @ y.
    # Extract A by evaluating f at basis vectors: n calls total.
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            break

    # Compute matrix exponential: y(t1) = expm(dt * A) @ y0
    # Use scipy's expm which uses Pade approximation + scaling/squaring
    # This is highly accurate for 15x15 matrices
    M = expm(dt * A)
    
    y_end = M @ y0

    return y_end
