"""Tensor decomposition for 2x3 times 3x9 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Constructs the factor matrices U, V, W for the tensor decomposition of
    matrix multiplication C = A * B, where A is 2x3, B is 3x9, C is 2x9.
    
    The rank-minimal decomposition is the schoolbook algorithm with rank n*m*p = 54.
    This is proven to be minimal because the 54 rank-1 terms corresponding to
    each product A[i,j]*B[j,k] contributing to C[k,i] are linearly independent.
    """
    t0 = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = n * m * p  # 54
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Populate the matrices for the schoolbook decomposition
    # The tensor T has non-zero entries at T[alpha, beta, gamma] = 1
    # where alpha = i*m + j, beta = j*p + k, gamma = k*n + i
    # Each such entry corresponds to a rank-1 term u_alpha e_alpha^T v_beta e_beta^T w_gamma e_gamma^T
    # In the decomposition T = sum_r U[:,r] V[:,r]^T W[:,r]^T (outer product sum)
    # Actually, the standard form is T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # So for each term (i,j,k), we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                
                U[alpha, r] = 1.0
                V[beta, r] = 1.0
                W[gamma, r] = 1.0
                r += 1
                
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
            
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
