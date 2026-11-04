import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    # ... (keep existing) ...
    pass

def schoolbook_factors(n, m, p):
    # ... (keep existing) ...
    pass

def rank_2_3x2x3_factors():
    n, m, p = 3, 2, 3
    rank = 2
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Term 1: A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0, 0] = 1.0
            V[0 * p + k, 0] = 1.0
            W[k * n + i, 0] = 1.0
            
    # Term 2: A[i,1] * B[1,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 1, 1] = 1.0
            V[1 * p + k, 1] = 1.0
            W[k * n + i, 1] = 1.0
            
    return U, V, W

def build_rank_26_3x3x3():
    # Returns U, V, W for 3x3x3 with rank 26
    n, m, p = 3, 3, 3
    n_sizes = [2, 1]
    m_sizes = [2, 1]
    p_sizes = [2, 1]
    
    # Reuse build_block_decomposition logic but for 3x3x3
    # Or just call it
    U, V, W, _ = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    return U, V, W

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    # ... (keep existing) ...
    pass

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    # ... (keep existing) ...
    pass

def schoolbook_lists(n, m, p):
    # ... (keep existing) ...
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # New strategy:
    # n=3, m=[3,2], p=[3,3]
    n_sizes = [3]
    m_sizes = [3, 2]
    p_sizes = [3, 3]
    
    assert sum(n_sizes) == n
    assert sum(m_sizes) == m
    assert sum(p_sizes) == p
    
    # We need a custom builder because blocks are not all same size type
    # Block (3,3,3) uses rank 26 logic
    # Block (3,2,3) uses rank 2 logic
    
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s
    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s
    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s

    # Calculate total rank
    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                if ni == 3 and mj == 3 and pk == 3:
                    total_rank += 26
                elif ni == 3 and mj == 2 and pk == 3:
                    total_rank += 2
                else:
                    total_rank += ni * mj * pk

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 3 and mj == 3 and pk == 3:
                    U_block, V_block, W_block = build_rank_26_3x3x3()
                elif ni == 3 and mj == 2 and pk == 3:
                    U_block, V_block, W_block = rank_2_3x2x3_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    if verify_decomposition(U_full, V_full, W_full, n, m, p, atol=1e-9):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
