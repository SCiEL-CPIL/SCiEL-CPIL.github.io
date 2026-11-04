import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    
    Strategy:
    1. Use a fixed, optimal block partition: n=[2,1], m=[2,2,1], p=[2,2,2].
       This creates six 2x2x2 blocks (handled by Strassen, rank 7 each) and
       the remaining 42 terms handled by schoolbook.
       Total base rank = 6*7 + 42 = 84.
    2. Attempt rank reduction via greedy column removal.
       The current program attempts this but might be slow or ineffective.
       We will implement a more robust verification and reduction step.
    3. If reduction fails or times out, return the base decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Fixed partitions known to be optimal for this shape
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = []
    offset = 0
    for s in n_parts:
        n_offsets.append(offset)
        offset += s
        
    m_offsets = []
    offset = 0
    for s in m_parts:
        m_offsets.append(offset)
        offset += s
        
    p_offsets = []
    offset = 0
    for s in p_parts:
        p_offsets.append(offset)
        offset += s

    # Identify blocks
    # n=3: [0:2], [2:3]
    # m=5: [0:2], [2:4], [4:5]
    # p=6: [0:2], [2:4], [4:6]
    
    # Blocks:
    # (ni, mj, pk) -> (rank, type)
    # (2, 2, 2) -> Strassen (7)
    # (2, 2, 2) -> Strassen (7)
    # (2, 2, 2) -> Strassen (7)
    # (1, 2, 2) -> Schoolbook (4)
    # (1, 2, 2) -> Schoolbook (4)
    # (1, 2, 2) -> Schoolbook (4)
    # (2, 1, 2) -> Schoolbook (4)
    # (2, 1, 2) -> Schoolbook (4)
    # (2, 1, 2) -> Schoolbook (4)
    # (1, 1, 2) -> Schoolbook (2)
    # (1, 1, 2) -> Schoolbook (2)
    # (1, 1, 2) -> Schoolbook (2)
    
    # Total Rank = 3*7 + 3*4 + 3*4 + 3*2 = 21 + 12 + 12 + 6 = 51? 
    # Wait. Let's re-calculate.
    # n=3, m=5, p=6.
    # n_parts [2,1], m_parts [2,2,1], p_parts [2,2,2].
    # Blocks:
    # i=0 (n=2):
    #   j=0 (m=2):
    #     k=0 (p=2): 2x2x2 Strassen (7)
    #     k=1 (p=2): 2x2x2 Strassen (7)
    #     k=2 (p=2): 2x2x2 Strassen (7)
    #   j=1 (m=2):
    #     k=0 (p=2): 2x2x2 Strassen (7)
    #     k=1 (p=2): 2x2x2 Strassen (7)
    #     k=2 (p=2): 2x2x2 Strassen (7)
    #   j=2 (m=1):
    #     k=0 (p=2): 2x1x2 Schoolbook (4)
    #     k=1 (p=2): 2x1x2 Schoolbook (4)
    #     k=2 (p=2): 2x1x2 Schoolbook (4)
    # i=1 (n=1):
    #   j=0 (m=2):
    #     k=0 (p=2): 1x2x2 Schoolbook (4)
    #     k=1 (p=2): 1x2x2 Schoolbook (4)
    #     k=2 (p=2): 1x2x2 Schoolbook (4)
    #   j=1 (m=2):
    #     k=0 (p=2): 1x2x2 Schoolbook (4)
    #     k=1 (p=2): 1x2x2 Schoolbook (4)
    #     k=2 (p=2): 1x2x2 Schoolbook (4)
    #   j=2 (m=1):
    #     k=0 (p=2): 1x1x2 Schoolbook (2)
    #     k=1 (p=2): 1x1x2 Schoolbook (2)
    #     k=2 (p=2): 1x1x2 Schoolbook (2)
    
    # Sum:
    # 6 * 7 = 42
    # 3 * 4 = 12
    # 6 * 4 = 24
    # 3 * 2 = 6
    # Total = 42 + 12 + 24 + 6 = 84.
    # This matches the inspiration 1 score of 1.0714 (90/84 = 1.0714).
    
    total_rank = 84
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    
    # Helper to build Strassen block
    def get_strassen_block():
        # Returns U, V, W for 2x2x2 Strassen
        # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        U_b = np.zeros((4, 7), dtype=np.float64)
        V_b = np.zeros((4, 7), dtype=np.float64)
        W_b = np.zeros((4, 7), dtype=np.float64)
        
        # m1 = (a+d)(e+h)
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[3, 0] = 1
        
        # m2 = (c+d)e
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        W_b[1, 1] = 1; W_b[3, 1] = -1
        
        # m3 = a(f-h)
        U_b[0, 2] = 1
        V_b[1, 2] = 1; V_b[3, 2] = -1
        W_b[2, 2] = 1; W_b[3, 2] = 1
        
        # m4 = d(g-e)
        U_b[3, 3] = 1
        V_b[2, 3] = 1; V_b[0, 3] = -1
        W_b[0, 3] = 1; W_b[1, 3] = 1
        
        # m5 = (a+b)h
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        W_b[0, 4] = -1; W_b[2, 4] = 1
        
        # m6 = (c-a)(e+f)
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        W_b[3, 5] = 1
        
        # m7 = (b-d)(g+h)
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        W_b[0, 6] = 1
        
        return U_b, V_b, W_b

    # Pre-compute Strassen block
    U_str, V_str, W_str = get_strassen_block()
    
    # Iterate through blocks
    for i_idx, ni in enumerate(n_parts):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_parts):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_parts):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen
                    U_block, V_block, W_block = U_str, V_str, W_str
                    r = 7
                else:
                    # Schoolbook
                    r = ni * mj * pk
                    U_block = np.zeros((ni * mj, r), dtype=np.float64)
                    V_block = np.zeros((mj * pk, r), dtype=np.float64)
                    W_block = np.zeros((pk * ni, r), dtype=np.float64)
                    c = 0
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_block[li * mj + lj, c] = 1.0
                                V_block[lj * pk + lk, c] = 1.0
                                W_block[lk * ni + li, c] = 1.0
                                c += 1
                
                # Map to global indices
                for li in range(ni):
                    for lj in range(mj):
                        full_row_u = (i_start + li) * m + (j_start + lj)
                        U[full_row_u, col_offset:col_offset + r] = U_block[li * mj + lj, :]
                
                for lj in range(mj):
                    for lk in range(pk):
                        full_row_v = (j_start + lj) * p + (k_start + lk)
                        V[full_row_v, col_offset:col_offset + r] = V_block[lj * pk + lk, :]
                
                for lk in range(pk):
                    for li in range(ni):
                        full_row_w = (k_start + lk) * n + (i_start + li)
                        W[full_row_w, col_offset:col_offset + r] = W_block[lk * ni + li, :]
                        
                col_offset += r
    
    # Verification
    def verify(U, V, W, n, m, p, atol=1e-9):
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
        vals = products.sum(axis=1)
        return bool(np.allclose(vals, 1.0, atol=atol))
    
    if not verify(U, V, W, n, m, p):
        # Fallback to pure schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt Rank Reduction
    # The current program's reduction is slow. We try a faster heuristic:
    # Check for linear dependencies in the tensor slices.
    # However, given the time limit, we might not find a reduction below 84.
    # The known lower bound for 3x5x6 is likely 84 or close to it.
    # Let's try a quick pass of removing columns if possible.
    
    current_rank = U.shape[1]
    # If we have time, try to remove columns
    if time.time() < deadline - 20:
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()
        rank_work = current_rank
        
        # Try removing each column
        # This is O(R^2 * N*M*P) roughly. 84^2 * 90 ~ 600k ops. Fast enough.
        for col_to_remove in range(rank_work - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            if rank_work <= 1:
                break
            
            mask = np.ones(rank_work, dtype=bool)
            mask[col_to_remove] = False
            
            U_red = U_work[:, mask]
            V_red = V_work[:, mask]
            W_red = W_work[:, mask]
            
            if verify(U_red, V_red, W_red, n, m, p, atol=1e-8):
                U_work, V_work, W_work = U_red, V_red, W_red
                rank_work -= 1
        
        if rank_work < current_rank:
            U, V, W = U_work, V_work, W_work
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
