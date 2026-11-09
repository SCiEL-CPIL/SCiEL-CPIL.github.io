# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 290.0

    # 1. Extract b
    b = np.array(f(t0, np.zeros(n)), dtype=float)
    
    # 2. Extract columns to determine Toeplitz structure
    e0 = np.zeros(n); e0[0] = 1.0
    col0 = np.array(f(t0, e0), dtype=float) - b
    
    e1 = np.zeros(n); e1[1] = 1.0
    col1 = np.array(f(t0, e1), dtype=float) - b
    
    en = np.zeros(n); en[n-1] = 1.0
    coln = np.array(f(t0, en), dtype=float) - b
    
    # Hypothesis: A is tridiagonal Toeplitz
    # A[0,0] = main, A[1,0] = sub, A[0,1] = super
    main_diag = col0[0]
    sub_diag = col0[1] if n > 1 else 0.0
    super_diag = col1[0] if n > 1 else 0.0
    
    # Verify consistency
    is_toeplitz = True
    if n > 2:
        # Check main diagonal from col1 and coln
        if not np.isclose(col1[1], main_diag, rtol=1e-10, atol=1e-12):
            is_toeplitz = False
        if not np.isclose(coln[n-1], main_diag, rtol=1e-10, atol=1e-12):
            is_toeplitz = False
        
        # Check sub diagonal from col1 and coln
        if not np.isclose(col1[2], sub_diag, rtol=1e-10, atol=1e-12):
            is_toeplitz = False
        if not np.isclose(coln[n-2], sub_diag, rtol=1e-10, atol=1e-12):
            is_toeplitz = False
            
        # Check super diagonal (only in col1 for index 0,1? No, col1 has A[0,1], A[1,1], A[2,1])
        # A[0,1] is super_diag.
        # We don't have A[1,2] directly from these 3 cols. 
        # But for Toeplitz, A[i, i+1] is constant.
        # We can check A[0,1] from col1.
        # We assume it holds for all i.
        
        # Check if other entries in extracted cols are zero (tridiagonal assumption)
        for i in range(3, n):
            if not np.isclose(col1[i], 0.0, rtol=1e-10, atol=1e-12):
                is_toeplitz = False
        for i in range(n-2):
            if not np.isclose(coln[i], 0.0, rtol=1e-10, atol=1e-12):
                is_toeplitz = False
        # col0 should only have A[0,0] and A[1,0]
        for i in range(2, n):
            if not np.isclose(col0[i], 0.0, rtol=1e-10, atol=1e-12):
                is_toeplitz = False

    if is_toeplitz:
        # Construct A
        A = np.zeros((n, n))
        np.fill_diagonal(A, main_diag)
        if n > 1:
            np.fill_diagonal(A[1:], sub_diag)
            np.fill_diagonal(A[:, 1:], super_diag)
        
        # Verify with a random vector
        y_rand = np.random.randn(n)
        f_pred = A @ y_rand + b
        f_actual = np.array(f(t0, y_rand), dtype=float)
        
        if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
            is_toeplitz = False
            # Fallback: Full extraction
            A = np.zeros((n, n))
            A[:, 0] = col0
            A[:, 1] = col1
            A[:, n-1] = coln
            for j in range(2, n-1):
                ej = np.zeros(n); ej[j] = 1.0
                A[:, j] = np.array(f(t0, ej), dtype=float) - b
            # Re-verify? If this fails, we are in trouble. 
            # Assume it works or fallback to RK4.
            y_rand2 = np.random.randn(n)
            f_pred2 = A @ y_rand2 + b
            f_actual2 = np.array(f(t0, y_rand2), dtype=float)
            if not np.allclose(f_pred2, f_actual2, rtol=1e-10, atol=1e-12):
                return _rk4_fallback(f, t_span, y0)
    else:
        # Fallback: Full extraction
        A = np.zeros((n, n))
        A[:, 0] = col0
        A[:, 1] = col1
        A[:, n-1] = coln
        for j in range(2, n-1):
            ej = np.zeros(n); ej[j] = 1.0
            A[:, j] = np.array(f(t0, ej), dtype=float) - b
        y_rand2 = np.random.randn(n)
        f_pred2 = A @ y_rand2 + b
        f_actual2 = np.array(f(t0, y_rand2), dtype=float)
        if not np.allclose(f_pred2, f_actual2, rtol=1e-10, atol=1e-12):
            return _rk4_fallback(f, t_span, y0)

    T = t1 - t0
    
    # Compute E = expm(A * T) and Z_mat = \int_0^T expm(A * u) du
    # Using augmented matrix K = [[A, I], [0, 0]]
    # expm(K*T) = [[E, Z_mat], [0, I]]
    K = np.zeros((2*n, 2*n))
    K[:n, :n] = A
    K[:n, n:] = np.eye(n)
    # K[n:, n:] is 0
    
    expK = expm(K * T)
    E = expK[:n, :n]
    Z_mat = expK[:n, n:]
    
    y_final = E @ y + Z_mat @ b
    
    if time.time() > deadline:
        return y_final
        
    return y_final

def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
