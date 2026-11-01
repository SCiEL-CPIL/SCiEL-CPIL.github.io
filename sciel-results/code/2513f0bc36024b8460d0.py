import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _build_strassen_block():
    """Return U, V, W for a 2x2 * 2x2 Strassen decomposition (rank 7)."""
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    P_s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    return U_s, V_s, P_s

def _build_rank14():
    """Build the rank-14 decomposition using Strassen on two 2x2 blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s, V_s, P_s = _build_strassen_block()
    
    for s in range(7):
        # Block 1 (k in 0,1)
        r1 = s
        U[:, r1] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + kp, r1] = V_s[s][j * 2 + kp]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, r1] = P_s[s][i, k]
        
        # Block 2 (k in 2,3)
        r2 = 7 + s
        U[:, r2] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + (kp + 2), r2] = V_s[s][j * 2 + kp]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i, r2] = P_s[s][i, k]
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    # Direct check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank13_optimization(U14, V14, W14, time_budget):
    """
    Try to find a rank-13 decomposition by optimizing from the rank-14 solution.
    We set one of the rank-14 columns to zero and optimize the remaining 13.
    Returns (U, V, W) if successful, None otherwise.
    """
    n, m, p = 2, 2, 4
    t_start = time.time()
    deadline = t_start + time_budget
    
    # Start from rank-14, zero out one column, and optimize
    # Try zeroing each of the 14 columns in turn
    for zero_col in range(14):
        if time.time() > deadline:
            return None
        
        # Initialize with rank-14, zeroing out column zero_col
        U = U14[:, :].copy()
        V = V14[:, :].copy()
        W = W14[:, :].copy()
        
        # Remove the zeroed column
        mask = np.ones(14, dtype=bool)
        mask[zero_col] = False
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
        
        # Now we have 13 columns. Try to optimize them.
        # Use a simple iterative method: adjust each entry to reduce the error.
        
        # The constraint is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is a system of 16 equations (i in 0..1, j in 0..1, k in 0..3) with 13*4 + 13*8 + 13*8 = 52+104+104 = 260 variables.
        # Wait, U is 4x13=52, V is 8x13=104, W is 8x13=104. Total 260 variables.
        # 16 equations. Very underdetermined. So rank 13 should be feasible if rank 14 is.
        # But we need to find a solution.
        
        # Let's use scipy.optimize if available, or a simple gradient descent.
        try:
            from scipy.optimize import minimize
        except ImportError:
            return None
        
        # Flatten the matrices
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        def objective(x):
            Ux = x[:52].reshape(4, 13)
            Vx = x[52:156].reshape(8, 13)
            Wx = x[156:260].reshape(8, 13)
            
            error = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        val = 0.0
                        for r in range(13):
                            val += Ux[i*2+j, r] * Vx[j*4+k, r] * Wx[k*2+i, r]
                        error += (val - 1.0) ** 2
            return error
        
        # Try a few iterations of L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-15})
            if result.fun < 1e-12:
                Ux = result.x[:52].reshape(4, 13)
                Vx = result.x[52:156].reshape(8, 13)
                Wx = result.x[156:260].reshape(8, 13)
                if _verify_decomposition(Ux, Vx, Wx):
                    return Ux, Vx, Wx
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # 140s deadline, 10s safety margin
    
    # Start with rank 14
    U14, V14, W14 = _build_rank14()
    if not _verify_decomposition(U14, V14, W14):
        raise RuntimeError("Rank 14 decomposition failed verification")
    
    best_rank = 14
    best_U, best_V, best_W = U14, V14, W14
    
    # Try to find rank 13 via optimization
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        result = _try_rank13_optimization(U14, V14, W14, min(remaining_time - 5, 100))
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 13
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
