import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This is based on the Kronecker product of Strassen's rank-7 algorithm
    for 2x2 matrix multiplication.
    
    The decomposition is exact and uses integer coefficients.
    """
    n, m, p = DIMENSIONS
    rank = 49
    
    # Local Strassen coefficients for 2x2 multiplication
    # U_local[t][idx] where idx = u*2+v for A[u,v]
    U_local = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    # V_local[t][idx] where idx = u*2+v for B[u,v]
    V_local = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # W_local[t][idx] where idx = u*2+v for C[u,v]
    W_local = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # Outer Strassen block coefficients
    # O_A[s][block] where block = i*2+j
    O_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    O_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # O_C[block][s] where block = i*2+j
    O_C = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Precompute indices for speed
    # Global index for A: (2*i+u)*m + (2*j+v)
    # Global index for B: (2*k+kk)*p + (2*l+ll)
    # Global index for C: (2*c_col+cv)*n + (2*c_row+cu)  [W[k*n+i, r]]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    b_idx = i * 2 + j
                    coeff_a = O_A[s][b_idx]
                    if coeff_a == 0: continue
                    for u in range(2):
                        for v in range(2):
                            g_idx = (2 * i + u) * m + (2 * j + v)
                            l_coeff = U_local[t][u * 2 + v]
                            if l_coeff != 0:
                                U[g_idx][col] += coeff_a * l_coeff
                                
            # Fill V
            for k in range(2):
                for l in range(2):
                    b_idx = k * 2 + l
                    coeff_b = O_B[s][b_idx]
                    if coeff_b == 0: continue
                    for kk in range(2):
                        for ll in range(2):
                            g_idx = (2 * k + kk) * p + (2 * l + ll)
                            l_coeff = V_local[t][kk * 2 + ll]
                            if l_coeff != 0:
                                V[g_idx][col] += coeff_b * l_coeff
                                
            # Fill W
            for c_row in range(2):
                for c_col in range(2):
                    b_idx = c_row * 2 + c_col
                    coeff_c = O_C[b_idx][s]
                    if coeff_c == 0: continue
                    for cu in range(2):
                        for cv in range(2):
                            g_idx = (c_col * 2 + cv) * n + (c_row * 2 + cu)
                            l_coeff = W_local[t][cu * 2 + cv]
                            if l_coeff != 0:
                                W[g_idx][col] += coeff_c * l_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
