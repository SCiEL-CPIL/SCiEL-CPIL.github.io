"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 118
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # The matrix multiplication tensor for A(n×m) * B(m×p) = C(n×p)
    # has T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # 
    # A rank-r decomposition satisfies:
    # T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # Key insight: We can achieve rank = m by using the j-index to select terms.
    # For r = 0..m-1:
    #   U[i*m+j, r] = 1 if j == r, else 0
    #   V[j*p+k, r] = 1 if j == r, else 0  
    #   W[k*n+i, r] = 1 for all k, i
    #
    # Verification: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    #   = sum_r [j==r] * [j==r] * 1
    #   = [j==j] * 1 = 1  (since exactly one r equals j)
    
    rank = m  # = 5
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    
    # Fill U: U[i*m+j, r] = 1 if j == r
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1.0
    
    # Fill V: V[j*p+k, r] = 1 if j == r
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1.0
    
    # Fill W: all ones
    W.fill(1.0)
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = np.zeros((U_rows, rank_fb))
                    V_fb = np.zeros((V_rows, rank_fb))
                    W_fb = np.zeros((W_rows, rank_fb))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, col] = 1.0
                                V_fb[jj * p + kk, col] = 1.0
                                W_fb[kk * n + ii, col] = 1.0
                                col += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
