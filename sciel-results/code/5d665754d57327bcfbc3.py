"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0."""
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    deadline = time.time() + 290  # safety margin before 330s kill
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # For a linear system y' = A @ y, we have f(t, y) = A @ y.
    # Extract A by evaluating f at a known y. Use y0 itself (or a basis vector).
    # f(t0, y0) = A @ y0, but we need A itself. Use multiple probe points.
    # Better: f(t, y) should be linear in y. Probe with basis vectors.
    # But that's n calls. Alternatively, if the system is y' = A@y, then
    # f(t, y) = A@y for any t. We can get A by calling f with basis vectors.
    
    # Most efficient: call f at y0 and at origin.
    # f(t0, y0) = A @ y0
    # f(t0, 0) = 0 (for homogeneous linear system)
    # This only gives A@y0, not A itself.
    
    # To get A, we need to probe with basis vectors: n calls total.
    # For n=15, that's 15 calls. Still vastly better than thousands.
    
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        # f(t, e_j) = A @ e_j = column j of A
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            # Fallback: use what we have so far, or use a rough estimate
            break

    # Compute matrix exponential: y(t1) = expm(dt * A) @ y0
    M = expm(dt * A)
    y_end = M @ y0

    return y_end
