import numpy as np
import time
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)

    # Rank 17 decomposition for 2x3x3 matrix multiplication
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))

    # Indices:
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j (6 entries)
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k (9 entries)
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (6 entries) [Trace convention W[k*n+i, r]]

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1

    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1

    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1

    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1

    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1

    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1

    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1

    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    # Verify the initial rank-17 solution
    if not verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 17
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # Strategy: Try to find a rank-16 decomposition by removing one term and solving for the rest
    # We use scipy.optimize.least_squares to find the optimal remaining terms
    
    def try_rank16(remove_idx):
        if time.time() > deadline - 10:
            return None
        
        # Keep all columns except remove_idx
        keep = [r for r in range(17) if r != remove_idx]
        U_init = U[:, keep]
        V_init = V[:, keep]
        W_init = W[:, keep]
        
        # Flatten the initial guess
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Define the residual function
        def residual(x):
            if time.time() > deadline - 5:
                return None # Signal to stop
            U_flat = x[:n*m*16]
            V_flat = x[n*m*16:n*m*16 + m*p*16]
            W_flat = x[n*m*16 + m*p*16:]
            
            U_mat = U_flat.reshape(n*m, 16)
            V_mat = V_flat.reshape(m*p, 16)
            W_mat = W_flat.reshape(p*n, 16)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)

        try:
            result = least_squares(residual, x0, method='lm', max_nfev=1000)
            if result.success and result.cost < 1e-18:
                # Reconstruct matrices
                U_sol = result.x[:n*m*16].reshape(n*m, 16)
                V_sol = result.x[n*m*16:n*m*16 + m*p*16].reshape(m*p, 16)
                W_sol = result.x[n*m*16 + m*p*16:].reshape(p*n, 16)
                
                if verify(U_sol, V_sol, W_sol, n, m, p):
                    return (U_sol, V_sol, W_sol)
        except Exception:
            pass
        return None

    # Try removing each term
    for i in range(17):
        if time.time() > deadline - 5:
            break
        sol = try_rank16(i)
        if sol is not None:
            best_rank = 16
            best_U, best_V, best_W = sol
            break

    # If we found rank 16, return it
    if best_rank == 16:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
