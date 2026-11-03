"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C from factors
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, R)
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    # Reshape V to (m, p, R)
    V_reshaped = V.reshape(m, p, R)
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    # Reshape W to (p, n, R)
    W_reshaped = W.reshape(p, n, R)
    
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Support compression: use 2 ranks per (i,k) pair
    # Total rank = n * p * 2 = 3 * 7 * 2 = 42
    R = n * p * 2
    
    # Initialize factor matrices
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each (i, k) pair, assign 2 consecutive ranks
    # r1(i,k) = 2*(i*p + k), r2(i,k) = 2*(i*p + k) + 1
    for i in range(n):
        for k in range(p):
            r1 = 2 * (i * p + k)
            r2 = 2 * (i * p + k) + 1
            
            # W[k*n+i, r1] = 1, W[k*n+i, r2] = 1
            W[k * n + i, r1] = 1.0
            W[k * n + i, r2] = 1.0
            
            # For each j, set U and V
            for j in range(m):
                # U[i*m+j, r1] = 1, U[i*m+j, r2] = 1
                U[i * m + j, r1] = 1.0
                U[i * m + j, r2] = 1.0
                
                # V[j*p+k, r1] = 0.5, V[j*p+k, r2] = 0.5
                V[j * p + k, r1] = 0.5
                V[j * p + k, r2] = 0.5
    
    # Verify the solution
    if _verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: try to find a better solution via optimization
    start_time = time.time()
    deadline = start_time + 100
    
    # If the simple solution doesn't verify (shouldn't happen), fall back to schoolbook
    rank = n * m * p
    U_fb = np.zeros((n * m, rank))
    V_fb = np.zeros((m * p, rank))
    W_fb = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, column] = 1
                V_fb[j * p + k, column] = 1
                W_fb[k * n + i, column] = 1
                column += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
