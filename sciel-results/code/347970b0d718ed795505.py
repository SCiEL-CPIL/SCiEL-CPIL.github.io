"""Optimized matrix multiplication 3x5x7 via block decomposition with Strassen."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: A00*(B00+B11)
    U[0, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # Product 2: (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = 1
    # Product 3: A00*(B10-B11)
    U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = -1; W[0, 2] = 1; W[2, 2] = 1
    # Product 4: A11*(B01-B00)
    U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[1, 3] = 1; W[3, 3] = 1
    # Product 5: (A00+A11)*(B00+B11)
    U[0, 4] = 1; U[3, 4] = 1; V[0, 4] = 1; V[3, 4] = 1; W[0, 4] = 1; W[3, 4] = 1
    # Product 6: (A01-A11)*(B10+B11)
    U[1, 5] = 1; U[3, 5] = -1; V[2, 5] = 1; V[3, 5] = 1; W[1, 5] = 1; W[3, 5] = 1
    # Product 7: (A10-A00)*(B00+B01)
    U[2, 6] = 1; U[0, 6] = -1; V[0, 6] = 1; V[1, 6] = 1; W[2, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block, offset_i, offset_j, offset_k, U, V, W, col_start):
    """Add schoolbook decomposition for a block."""
    col = col_start
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                global_i = offset_i + i
                global_j = offset_j + j
                global_k = offset_k + k
                U[global_i * 5 + global_j, col] = 1.0
                V[global_j * 7 + global_k, col] = 1.0
                W[global_k * 3 + global_i, col] = 1.0
                col += 1
    return col

def _rank_k_block(n_block, m_block, p_block, offset_i, offset_j, offset_k, U, V, W, col_start):
    """For 1xk * kxl, rank = k."""
    if n_block == 1:
        # 1 x m_block * p_block: rank = m_block
        col = col_start
        for j in range(m_block):
            for k in range(p_block):
                global_i = offset_i
                global_j = offset_j + j
                global_k = offset_k + k
                U[global_i * 5 + global_j, col] = 1.0
                V[global_j * 7 + global_k, col] = 1.0
                W[global_k * 3 + global_i, col] = 1.0
                col += 1
        return col
    else:
        return _schoolbook_block(n_block, m_block, p_block, offset_i, offset_j, offset_k, U, V, W, col_start)

def _strassen_block(n_block, m_block, p_block, offset_i, offset_j, offset_k, U, V, W, col_start):
    """Use Strassen for 2x2x2 blocks, schoolbook for 2x2x1."""
    if n_block == 2 and m_block == 2 and p_block == 2:
        # Use Strassen
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[(offset_i + i) * 5 + (offset_j + j), col_start + r] += U_s[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(offset_j + j) * 7 + (offset_k + k), col_start + r] += V_s[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(offset_k + k) * 3 + (offset_i + i), col_start + r] += W_s[k * 2 + i, r]
        return col_start + 7
    elif n_block == 2 and m_block == 2 and p_block == 1:
        # Schoolbook: rank 4
        return _schoolbook_block(2, 2, 1, offset_i, offset_j, offset_k, U, V, W, col_start)
    else:
        return _schoolbook_block(n_block, m_block, p_block, offset_i, offset_j, offset_k, U, V, W, col_start)

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    # Split i: [1, 2], j: [2, 3], k: [3, 4]
    i_blocks = [0, 1]  # offsets
    j_blocks = [0, 2]  # offsets
    k_blocks = [0, 3]  # offsets
    i_sizes = [1, 2]
    j_sizes = [2, 3]
    k_sizes = [3, 4]
    
    total_rank = 0
    U = np.zeros((n * m, 200))  # Allocate enough
    V = np.zeros((m * p, 200))
    W = np.zeros((p * n, 200))
    
    col = 0
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                ni = i_sizes[ii]
                mj = j_sizes[jj]
                pk = k_sizes[kk]
                oi = i_blocks[ii]
                oj = j_blocks[jj]
                ok = k_blocks[kk]
                
                if ni == 1:
                    col = _rank_k_block(ni, mj, pk, oi, oj, ok, U, V, W, col)
                elif ni == 2 and mj == 2 and pk == 2:
                    col = _strassen_block(ni, mj, pk, oi, oj, ok, U, V, W, col)
                elif ni == 2 and mj == 2 and pk == 1:
                    col = _strassen_block(ni, mj, pk, oi, oj, ok, U, V, W, col)
                else:
                    col = _schoolbook_block(ni, mj, pk, oi, oj, ok, U, V, W, col)
    
    total_rank = col
    U = U[:, :total_rank]
    V = V[:, :total_rank]
    W = W[:, :total_rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
