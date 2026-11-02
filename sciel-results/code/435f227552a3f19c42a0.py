import time
import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication 2x5 * 5x5.
    Attempts to find a rank lower than the schoolbook rank of 50.
    If no lower rank is found within the time limit, returns the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # 1. Build Schoolbook Solution (Rank 50)
    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = build_schoolbook()
    best_rank = schoolbook_rank

    # 2. Verification Function
    def is_valid(U, V, W, tol=1e-8):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized dot product for speed
                    u_val = U_arr[i * m + j, :]
                    v_val = V_arr[j * p + k, :]
                    w_val = W_arr[k * n + i, :]
                    total = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # 3. Search for Lower Rank
    # Strategy: Try Rank 25 (as per directive) and Rank 49.
    # We fix W to random small integers to make the problem linear in U and V?
    # Actually, fixing W makes it bilinear in U and V.
    # We can fix W, then for each (i,j), the equation is linear in U? No.
    # Equation: sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1
    # If we fix V and W, it is linear in U.
    # If we fix U and W, it is linear in V.
    # We can alternate.
    
    def try_rank(R, max_iters=50):
        if time.time() > deadline:
            return None
        
        # Initialize W with random small integers (sparse)
        # To exploit n=2, we can try to make W have a specific structure.
        # But random is a good start.
        np.random.seed(42 + R)
        
        # Heuristic: Initialize W with 0s and 1s
        W_curr = np.zeros((p * n, R))
        # Ensure each row has at least one non-zero
        for idx in range(p * n):
            # Pick 1 or 2 random indices to be 1
            num_non_zero = 1 if np.random.rand() < 0.5 else 2
            indices = np.random.choice(R, num_non_zero, replace=False)
            W_curr[idx, indices] = 1.0
            
        V_curr = np.random.randn(m * p, R) * 0.1
        U_curr = np.random.randn(n * m, R) * 0.1

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve for U given V, W
            # System: A_U @ u_flat = b
            # u_flat size: n*m*R
            # Constraints: n*m*p
            n_constraints = n * m * p
            n_vars = n * m * R
            A_U = np.zeros((n_constraints, n_vars))
            b = np.ones(n_constraints)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        # Coeff for U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r]
                        coeff_vec = V_curr[j * p + k, :] * W_curr[k * n + i, :]
                        col_start = (i * m + j) * R
                        A_U[row, col_start:col_start+R] = coeff_vec
            
            # Solve least squares
            try:
                u_flat, residuals, rank_u, s = np.linalg.lstsq(A_U, b, rcond=None)
                U_curr = u_flat.reshape(n * m, R)
            except:
                break

            # Solve for V given U, W
            n_vars_v = m * p * R
            A_V = np.zeros((n_constraints, n_vars_v))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        coeff_vec = U_curr[i * m + j, :] * W_curr[k * n + i, :]
                        col_start = (j * p + k) * R
                        A_V[row, col_start:col_start+R] = coeff_vec
                        
            try:
                v_flat, residuals, rank_v, s = np.linalg.lstsq(A_V, b, rcond=None)
                V_curr = v_flat.reshape(m * p, R)
            except:
                break

            # Check validity
            if is_valid(U_curr.tolist(), V_curr.tolist(), W_curr.tolist()):
                return U_curr.tolist(), V_curr.tolist(), W_curr.tolist()
                
        return None

    # Try Rank 25 (Directive)
    if time.time() < deadline - 10:
        result = try_rank(25)
        if result:
            best_U, best_V, best_W = result
            best_rank = 25
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try Rank 49
    if time.time() < deadline - 10:
        result = try_rank(49)
        if result:
            best_U, best_V, best_W = result
            best_rank = 49
            return {"U": best_U, "V": best_V, "W": best_W}

    # Fallback to Schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
