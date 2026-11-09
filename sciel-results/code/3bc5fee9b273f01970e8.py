import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 140.0  # 10s buffer before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
        
    # Scale x to z in [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    x_center = (x_max + x_min) / 2.0
    x_half_range = (x_max - x_min) / 2.0
    
    if x_half_range < 1e-9:
        x_half_range = 1.0
        
    z = (x - x_center) / x_half_range
    
    # Precompute Legendre polynomials P_0 to P_5 for z
    # P_0 = 1
    # P_1 = z
    # P_2 = 0.5 * (3*z^2 - 1)
    # P_3 = 0.5 * (5*z^3 - 3*z)
    # P_4 = 0.125 * (35*z^4 - 30*z^2 + 3)
    # P_5 = 0.125 * (63*z^5 - 70*z^3 + 15*z)
    
    L = np.zeros((n, 6))
    L[:, 0] = 1.0
    L[:, 1] = z
    L[:, 2] = 0.5 * (3.0 * z**2 - 1.0)
    L[:, 3] = 0.5 * (5.0 * z**3 - 3.0 * z)
    L[:, 4] = 0.125 * (35.0 * z**4 - 30.0 * z**2 + 3.0)
    L[:, 5] = 0.125 * (63.0 * z**5 - 70.0 * z**3 + 15.0 * z)
    
    # Transformation matrix from Legendre coeffs (a) to monomial coeffs in z (c_z)
    # P_k(z) = sum_{j=0}^k M[k,j] z^j
    # We need c_z[j] = sum_{k=j}^5 a[k] * M[k,j]
    # Let's build M such that P_k(z) = sum_j M[k,j] z^j
    M = np.zeros((6, 6))
    M[0, 0] = 1.0
    M[1, 1] = 1.0
    M[2, 0] = -0.5
    M[2, 2] = 1.5
    M[3, 1] = -1.5
    M[3, 3] = 2.5
    M[4, 0] = 0.375
    M[4, 2] = -3.75
    M[4, 4] = 4.375
    M[5, 1] = 1.875
    M[5, 3] = -8.75
    M[5, 5] = 7.875
    
    # c_z = a @ M (where a is row vector of Legendre coeffs, M is 6x6 matrix)
    # Wait, a is shape (6,). c_z[j] = sum_k a[k] M[k,j]. So c_z = a @ M.
    
    # Transformation from z to x
    # z = (x - x_center) / x_half_range = alpha * x + beta
    alpha = 1.0 / x_half_range
    beta = -x_center / x_half_range
    
    # c_x[j] = sum_{k=j}^5 c_z[k] * comb(k, j) * alpha^j * beta^(k-j)
    
    def convert_z_to_x(c_z):
        c_x = np.zeros(6)
        for j in range(6):
            c_j = 0.0
            for k in range(j, 6):
                term = c_z[k] * comb(k, j) * (alpha ** j) * (beta ** (k - j))
                c_j += term
            c_x[j] = c_j
        return c_x

    def evaluate_mse(c_x, x_eval, y_eval):
        # Evaluate polynomial sum c_i x^i
        # Use Horner's method for stability and speed
        y_pred = np.zeros_like(x_eval)
        for i in range(5, -1, -1):
            y_pred = y_pred * x_eval + c_x[i]
        mse = np.mean((y_pred - y_eval) ** 2)
        return mse

    # Grid of lambda values
    # Trace of L^T L is roughly n * sum(P_k^2). For Legendre, sum P_k^2 is ~ n.
    # So L^T L is ~ n * I.
    # We want lambda to be small relative to n.
    lam_grid = np.logspace(-10, 0, 15)
    
    # Split data for lambda selection
    # Use a simple random shuffle split
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    split = int(0.9 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    L_train = L[train_idx]
    y_train = y[train_idx]
    L_val = L[val_idx]
    y_val = y[val_idx]
    x_train = x[train_idx]
    x_val = x[val_idx]
    
    best_lam = lam_grid[0]
    best_val_mse = np.inf
    best_a = None
    
    # Precompute L^T L and L^T y for train
    LtL = L_train.T @ L_train
    Lty = L_train.T @ y_train
    
    for lam in lam_grid:
        A = LtL + lam * np.eye(6)
        try:
            a = np.linalg.solve(A, Lty)
        except np.linalg.LinAlgError:
            continue
            
        # Convert a to c_z
        c_z = a @ M
        
        # Convert c_z to c_x
        c_x = convert_z_to_x(c_z)
        
        # Evaluate on validation set
        mse = evaluate_mse(c_x, x_val, y_val)
        
        if mse < best_val_mse:
            best_val_mse = mse
            best_lam = lam
            best_a = a
            
        # Time check
        if time.time() - start_time > deadline - 10:
            break
            
    # Refit on entire dataset with best lambda
    LtL_full = L.T @ L
    Lty_full = L.T @ y
    A_full = LtL_full + best_lam * np.eye(6)
    
    try:
        a_final = np.linalg.solve(A_full, Lty_full)
    except np.linalg.LinAlgError:
        # Fallback to least squares if solve fails (unlikely with ridge)
        a_final = np.linalg.lstsq(L, y, rcond=None)[0]
        
    c_z_final = a_final @ M
    c_x_final = convert_z_to_x(c_z_final)
    
    return c_x_final
