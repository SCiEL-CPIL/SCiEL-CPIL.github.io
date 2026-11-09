import numpy as np
import scipy.linalg as la

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Extract the linear operator A
    # Assumption: f(t, y) = A @ y, with A constant in time
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        
    # Number of steps
    # Padé (15,15) is high order, so few steps are needed for high accuracy
    steps = 10
    h = (t1 - t0) / steps
    
    # Padé order
    N = 15
    
    # Compute Padé coefficients c_k
    # c_k = c_{k-1} * (N - k + 1) / (k * (2*N - k + 1))
    c = np.zeros(N + 1)
    c[0] = 1.0
    for k in range(1, N + 1):
        c[k] = c[k-1] * (N - k + 1) / (k * (2*N - k + 1))
        
    # Build M_L and M_R
    # M_L = sum c_k (-h A)^k
    # M_R = sum c_k (h A)^k
    A_power = np.eye(n)
    M_L = np.zeros((n, n))
    M_R = np.zeros((n, n))
    
    for k in range(N + 1):
        M_L += c[k] * ((-h) ** k) * A_power
        M_R += c[k] * (h ** k) * A_power
        if k < N:
            A_power = A_power @ A
            
    # Factorize M_L
    lu, piv = la.lu_factor(M_L)
    
    y_current = y.copy()
    for _ in range(steps):
        rhs = M_R @ y_current
        y_current = la.lu_solve((lu, piv), rhs)
        
    return y_current
