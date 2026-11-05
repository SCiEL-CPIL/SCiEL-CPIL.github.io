import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We want to find U, V, W with integer/half-integer entries.
    # We will search for a low rank R.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute indices
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    for R in range(24, 64):
        # Try a few random initializations
        for attempt in range(5):
            np.random.seed(123 + attempt)
            # Initialize with small random values
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Alternating minimization
            success = False
            for it in range(50):
                # Solve for U
                # For each (i, j), we have:
                # sum_r U[i*m+j, r] * (V[j*p+0:..., r] * W[0:..., r] * ...) ? No.
                # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
                # For fixed V, W, the equation for a specific (i, j) is:
                # For each k: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let A_r = V[j*p+k, r] * W[k*n+i, r].
                # Then sum_r U[i*m+j, r] * A_r = 1.
                # This is a linear system for U[i*m+j, :] given V, W and fixed i, j.
                
                # We can set up a large linear system for all U variables.
                # Number of equations: n * m * p = 63.
                # Number of unknowns: n * m * R = 9 * R.
                
                A_mat = np.zeros((n*m*p, n*m*R))
                b_vec = np.ones(n*m*p)
                
                # Fill A_mat
                # Row index: idx = (i * m + j) * p + k
                # Col index: u_idx = (i * m + j) * R + r
                # Value: V[j*p+k, r] * W[k*n+i, r]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i * m + j) * p + k
                            for r in range(R):
                                col = (i * m + j) * R + r
                                A_mat[row, col] = V[j*p+k, r] * W[k*n+i, r]
                
                # Solve least squares
                U_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U = U_sol.reshape(n*m, R)
                
                # Check residual
                # Reconstruct tensor
                T_recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            T_recon[i, j, k] = val
                err = np.sum((T_recon - 1)**2)
                
                if err < 1e-8:
                    success = True
                    break
                
                # Similarly for V
                A_mat = np.zeros((n*m*p, m*p*R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i * m + j) * p + k
                            for r in range(R):
                                col = (j * p + k) * R + r
                                A_mat[row, col] = U[i*m+j, r] * W[k*n+i, r]
                
                V_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V = V_sol.reshape(m*p, R)
                
                # Similarly for W
                A_mat = np.zeros((n*m*p, p*n*R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i * m + j) * p + k
                            for r in range(R):
                                col = (k * n + i) * R + r
                                A_mat[row, col] = U[i*m+j, r] * V[j*p+k, r]
                
                W_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_sol.reshape(p*n, R)
            
            if success:
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                T_recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                            T_recon[i, j, k] = val
                            
                if np.allclose(T_recon, 1, atol=1e-9):
                    best_rank = R
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    break
        
        if best_rank < n * m * p:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
