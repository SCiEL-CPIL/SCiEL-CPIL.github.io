DIMENSIONS = [2, 4, 8]

import time
_START_TIME = time.time()
_DEADLINE = 110  # seconds, safety margin for 120s limit

def _build_strassen_block():
    """
    Returns U, V, W entries for a single 2x2x2 Strassen decomposition.
    Indices:
    U: i*2+j (i in 0..1, j in 0..1) -> 4 entries
    V: j*2+k (j in 0..1, k in 0..1) -> 4 entries
    W: k*2+i (k in 0..1, i in 0..1) -> 4 entries (Trace convention: W[k*n+i])
    
    Strassen 7 products:
    m0 = (A00+A11)(B00+B11)
    m1 = (A10+A11)B00
    m2 = A00(B10-B11)
    m3 = A11(B01-B00)
    m4 = (A00+A01)B11
    m5 = (A10-A00)(B00+B01)
    m6 = (A01-A11)(B10+B11)
    
    C00 = m0 + m3 - m4 - m6
    C01 = m2 + m4
    C10 = m1 + m3
    C11 = m0 - m1 + m2 + m5
    """
    # We store non-zero (index, value) pairs for each rank component r in 0..6
    # U[r][idx] = coeff of A_idx in product r
    # V[r][idx] = coeff of B_idx in product r
    # W[r][idx] = coeff of C_idx in product r (added to result)
    
    U = [[], [], [], [], [], [], []]
    V = [[], [], [], [], [], [], []]
    W = [[], [], [], [], [], [], []]
    
    # r=0: m0
    # U: A00, A11 -> idx 0, 3
    U[0] = [(0, 1), (3, 1)]
    # V: B00, B11 -> idx 0, 3
    V[0] = [(0, 1), (3, 1)]
    # W: C00, C11 -> idx 0, 3 (k=0,i=0 -> 0; k=1,i=1 -> 3)
    W[0] = [(0, 1), (3, 1)]
    
    # r=1: m1
    # U: A10, A11 -> idx 2, 3
    U[1] = [(2, 1), (3, 1)]
    # V: B00 -> idx 0
    V[1] = [(0, 1)]
    # W: C10, C11 -> idx 1, 3 (k=0,i=1 -> 1; k=1,i=1 -> 3)
    W[1] = [(1, 1), (3, 1)]
    
    # r=2: m2
    # U: A00 -> idx 0
    U[2] = [(0, 1)]
    # V: B10, -B11 -> idx 2, 3
    V[2] = [(2, 1), (3, -1)]
    # W: C01, C11 -> idx 2, 3 (k=1,i=0 -> 2; k=1,i=1 -> 3)
    W[2] = [(2, 1), (3, 1)]
    
    # r=3: m3
    # U: A11 -> idx 3
    U[3] = [(3, 1)]
    # V: B01, -B00 -> idx 1, 0
    V[3] = [(1, 1), (0, -1)]
    # W: C00, C10 -> idx 0, 1 (k=0,i=0 -> 0; k=0,i=1 -> 1)
    W[3] = [(0, 1), (1, 1)]
    
    # r=4: m4
    # U: A00, A01 -> idx 0, 1
    U[4] = [(0, 1), (1, 1)]
    # V: B11 -> idx 3
    V[4] = [(3, 1)]
    # W: -C00, C01 -> idx 0, 2
    W[4] = [(0, -1), (2, 1)]
    
    # r=5: m5
    # U: A10, -A00 -> idx 2, 0
    U[5] = [(2, 1), (0, -1)]
    # V: B00, B01 -> idx 0, 1
    V[5] = [(0, 1), (1, 1)]
    # W: C11 -> idx 3
    W[5] = [(3, 1)]
    
    # r=6: m6
    # U: A01, -A11 -> idx 1, 3
    U[6] = [(1, 1), (3, -1)]
    # V: B10, B11 -> idx 2, 3
    V[6] = [(2, 1), (3, 1)]
    # W: -C00 -> idx 0
    W[6] = [(0, -1)]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 56
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-compute the Strassen block patterns
    sU, sV, sW = _build_strassen_block()
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            # Update U
            for idx, val in sU[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            # Update V
            for idx, val in sV[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            # Update W
            for idx, val in sW[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    # The 2x4x8 multiplication decomposes into 8 independent 2x2x2 multiplications.
    # We iterate through the 2 blocks of A (rows fixed, cols 0-1 and 2-3) 
    # and 4 blocks of B (cols fixed, k 0-1, 2-3, 4-5, 6-7).
    # i_offset is always 0 since A has only 2 rows.
    
    block_id = 0
    for a_col_block in [0, 2]:  # j_offset for A and B
        for b_col_block in [0, 2, 4, 6]:  # k_offset for B
            # A block is A[:, j_off:j_off+2]
            # B block is B[j_off:j_off+2, k_off:k_off+2]
            # Wait, B is 4x8. 
            # A is 2x4.
            # C is 2x8.
            # C[:, k:k+2] = A[:, j:j+2] * B[j:j+2, k:k+2]
            # This requires A's columns j..j+1 and B's rows j..j+1.
            # So for each pair of columns in A (which correspond to rows in B), we have 4 output blocks.
            
            # My loop structure above:
            # a_col_block corresponds to the row index in B and col index in A.
            # b_col_block corresponds to the col index in B (and thus C).
            
            add_block(block_id * 7, 0, a_col_block, b_col_block)
            block_id += 1
            
            # Check time
            if time.time() - _START_TIME > _DEADLINE:
                # Return whatever we have so far (should be complete if fast)
                break
        if time.time() - _START_TIME > _DEADLINE:
            break

    return {"U": U, "V": V, "W": W}
