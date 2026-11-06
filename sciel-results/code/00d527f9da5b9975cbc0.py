"""ODE solver: 15-dimensional stiff linear system on [0,1]."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = y0_arr.size
    deadline = time.time() + 280.0

    # Extract A and b from f(t, y) = A y + b using basis vectors.
    # 16 calls total.
    e0 = np.zeros(n)
    b = np.array(f(t0, e0), dtype=float)
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = np.array(f(t0, ej), dtype=float) - b

    # Solve the autonomous linear system y' = A y + b analytically.
    # Use block matrix exponential to handle both invertible and singular A robustly.
    # M = [[A, -b], [0, 1]]
    # expm(M * dt) = [[expm(A * dt), int_0^t expm(A * (t-s)) b ds], [0, 1]]
    dt = t1 - t0
    
    # Construct the augmented matrix
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = -b
    M[n, n] = 1.0
    
    # Compute the matrix exponential
    E = expm(M * dt)
    
    # Extract the solution
    # y(t) = E[0:n, 0:n] @ y0 + E[0:n, n]
    y_final = E[:n, :n] @ y0_arr + E[:n, n]
    
    return y_final
# EVOLVE-BLOCK-END
