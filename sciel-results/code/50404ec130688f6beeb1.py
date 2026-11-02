import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = n * m * p # 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Schoolbook decomposition
    # r = i*m + j
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1 (only if k is determined? No, V is j,k)
    # Wait, schoolbook: C_ik = sum_j A_ij B_jk
    # Tensor T_ijk = 1.
    # Decomposition: sum_r U_ijr V_jkr W_kir = 1.
    # Let r index the (i,j) pair?
    # If r = i*m + j, then U_ijr = delta_{r, i*m+j}.
    # V_jkr must be 1 for all k? No.
    # If V_jkr = delta_{j, j_r}, then W_kir must handle the rest.
    # Standard rank n*m*p decomposition:
    # r = (i, j, k).
    # U_ijr = 1 if r=(i,j,k).
    # V_jkr = 1 if r=(i,j,k).
    # W_kir = 1 if r=(i,j,k).
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
