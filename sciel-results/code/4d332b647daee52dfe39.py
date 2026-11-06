import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Determine if the system is linear (homogeneous) or affine
    # f(t, y) = A(t)y + b(t)
    # Evaluate at y=0 to get b(t0)
    y_zero = np.zeros(n)
    f_zero = f(t0, y_zero)
    
    # Check if b is negligible (linear homogeneous)
    # If f(t0, 0) is close to 0, assume homogeneous
    is_homogeneous = np.allclose(f_zero, 0, atol=1e-10)
    
    # We will use a step-by-step integration with matrix exponentials
    # This handles time-varying A(t) reasonably well with sufficient steps
    # For stiff problems, even few steps of expm are very accurate
    steps = 10
    h = dt / steps
    
    # Current state
    y = y0.copy()
    t = t0
    
    for _ in range(steps):
        # Evaluate A(t) and b(t) at current t
        # We need n calls for columns of A, and we already have b if we re-evaluate f_zero
        # To save calls, we can just evaluate f(t, e_i) and subtract f(t, 0)
        # But f(t, 0) changes if b(t) varies.
        # So we need 1 call for b(t) and n calls for A(t) columns.
        
        b_t = f(t, y_zero)
        
        A = np.empty((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            # f(t, e_j) = A(t) @ e_j + b(t) = A[:, j] + b(t)
            # So A[:, j] = f(t, e_j) - b(t)
            A[:, j] = f(t, e_j) - b_t
            
        if is_homogeneous and np.allclose(b_t, 0, atol=1e-10):
            # Homogeneous case: y' = A y
            # Solution: y(t+h) = expm(A*h) @ y
            M = scipy.linalg.expm(A * h)
            y = M @ y
        else:
            # Affine case: y' = A y + b
            # Augment: Y = [y; 1], Y' = [[A, b], [0, 0]] Y
            A_aug = np.zeros((n+1, n+1))
            A_aug[:n, :n] = A
            A_aug[:n, n] = b_t
            # Last row is 0
            
            y_aug = np.zeros(n+1)
            y_aug[:n] = y
            y_aug[n] = 1.0
            
            M_aug = scipy.linalg.expm(A_aug * h)
            y_aug_new = M_aug @ y_aug
            y = y_aug_new[:n]
            
        t += h
        
    return y
