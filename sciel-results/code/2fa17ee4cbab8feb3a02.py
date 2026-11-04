import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Construct schoolbook decomposition for n x m x p block.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank).
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Partition 5 = 2 + 2 + 1
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    total_rank = 0
    # First pass: calculate total rank to allocate memory
    for i_idx in range(3):
        for j_idx in range(3):
            for k_idx in range(3):
                ni = sizes[i_idx]
                mj = sizes[j_idx]
                pk = sizes[k_idx]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
                    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    U2, V2, W2 = _strassen_2x2()
    
    for i_idx in range(3):
        for j_idx in range(3):
            for k_idx in range(3):
                ni = sizes[i_idx]
                mj = sizes[j_idx]
                pk = sizes[k_idx]
                
                off_i = offsets[i_idx]
                off_j = offsets[j_idx]
                off_k = offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    # U2 is 4x7, V2 is 4x7, W2 is 4x7
                    # Map local indices to global
                    # Local i in [0,2), j in [0,2), k in [0,2)
                    # Global i = off_i + i, etc.
                    
                    # U rows: i*mj + j -> (off_i + i)*5 + (off_j + j)
                    # V rows: mj*pk + k -> (off_j + j)*5 + (off_k + k)
                    # W rows: pk*ni + i -> (off_k + k)*5 + (off_i + i)
                    
                    for c in range(7):
                        # U
                        for li in range(2):
                            for lj in range(2):
                                val = U2[li * 2 + lj, c]
                                if val != 0:
                                    gi = off_i + li
                                    gj = off_j + lj
                                    U[gi * 5 + gj, col_offset + c] = val
                        # V
                        for lj in range(2):
                            for lk in range(2):
                                val = V2[lj * 2 + lk, c]
                                if val != 0:
                                    gj = off_j + lj
                                    gk = off_k + lk
                                    V[gj * 5 + gk, col_offset + c] = val
                        # W
                        for lk in range(2):
                            for li in range(2):
                                val = W2[lk * 2 + li, c]
                                if val != 0:
                                    gk = off_k + lk
                                    gi = off_i + li
                                    W[gk * 5 + gi, col_offset + c] = val
                    col_offset += 7
                else:
                    # Use Schoolbook
                    Ub, Vb, Wb = _schoolbook_block(ni, mj, pk)
                    rank_b = Ub.shape[1]
                    
                    for c in range(rank_b):
                        # U
                        for li in range(ni):
                            for lj in range(mj):
                                val = Ub[li * mj + lj, c]
                                if val != 0:
                                    gi = off_i + li
                                    gj = off_j + lj
                                    U[gi * 5 + gj, col_offset + c] = val
                        # V
                        for lj in range(mj):
                            for lk in range(pk):
                                val = Vb[lj * pk + lk, c]
                                if val != 0:
                                    gj = off_j + lj
                                    gk = off_k + lk
                                    V[gj * 5 + gk, col_offset + c] = val
                        # W
                        for lk in range(pk):
                            for li in range(ni):
                                val = Wb[lk * ni + li, c]
                                if val != 0:
                                    gk = off_k + lk
                                    gi = off_i + li
                                    W[gk * 5 + gi, col_offset + c] = val
                    col_offset += rank_b
                    
    return {"U": U, "V": V, "W": W}
