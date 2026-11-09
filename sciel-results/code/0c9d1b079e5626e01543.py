"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 300.0  # Safety margin before 330s kill

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Probe for linear time-invariant structure: y' = A*y + b
    # b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)

    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)

    # Assume Linear Time-Invariant (LTI) system based on problem description
    # This saves 2 function calls compared to verifying linearity and time-invariance
    # Total calls: 1 (for b) + 15 (for A) = 16
    # Use augmented matrix approach for exact solution
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is all zeros

    y_aug0 = np.zeros(n + 1)
    y_aug0[:n] = y0
    y_aug0[n] = 1.0

    M = expm(A_aug * dt)
    y_aug = M @ y_aug0
    y_result = y_aug[:n].copy()

    return y_result
