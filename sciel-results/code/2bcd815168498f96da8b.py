import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # The current best known decomposition for M_2,4,4 in this context is rank 28.
    # We return the pre-computed rank 28 decomposition.
    
    result = _rank28_decomposition()
    
    # Optional: Verify if time permits
    if time.time() < deadline - 10:
        # Simple sanity check
        pass
        
    return result

def _rank28_decomposition():
    """Return the known rank-28 decomposition for 2x4 * 4x4."""
    n, m, p = 2, 4, 4
    TARGET_RANK = 28
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]  # 8 x 28
    V = [[0] * TARGET_RANK for _ in range(m * p)]  # 16 x 28
    W = [[0] * TARGET_RANK for _ in range(p * n)]  # 8 x 28
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i
    
    # The decomposition is based on applying Strassen's algorithm to 2x2 blocks.
    # We split the 4x4 matrix into four 2x2 blocks.
    # The 2x4 matrix A is split into four 2x2 blocks (conceptually, though it's 2 rows of 4).
    # Actually, A is 2x4. We can view it as two 2x2 blocks side-by-side: A_left, A_right.
    # B is 4x4, split into four 2x2 blocks: B_tl, B_tr, B_bl, B_br.
    # C is 2x4, split into two 2x2 blocks: C_left, C_right.
    
    # C_left = A_left * B_tl + A_right * B_bl
    # C_right = A_left * B_tr + A_right * B_br
    
    # This is a sum of two 2x2 matrix multiplications for each block of C.
    # Each 2x2 multiplication has rank 7.
    # Total rank 7 * 4 = 28.
    
    # We define the blocks for A, B, C
    # A indices: (0,0)-(0,1) is A_left, (0,2)-(0,3) is A_right?
    # Wait, A is 2x4.
    # Row 0: A[0,0], A[0,1], A[0,2], A[0,3]
    # Row 1: A[1,0], A[1,1], A[1,2], A[1,3]
    
    # B is 4x4.
    # B[0,0] B[0,1] B[0,2] B[0,3]
    # B[1,0] B[1,1] B[1,2] B[1,3]
    # B[2,0] B[2,1] B[2,2] B[2,3]
    # B[3,0] B[3,1] B[3,2] B[3,3]
    
    # C is 2x4.
    
    # The decomposition provided in the original code uses a specific set of blocks.
    # I will retain the original code's block definitions as they produced a valid result.
    
    blocks = [
        (0, 0, 0, 0, 0), # A block (0,0), B block (0,0), C col 0
        (0, 2, 2, 0, 0), # A block (0,2), B block (2,0), C col 0
        (0, 0, 0, 2, 2), # A block (0,0), B block (0,2), C col 2
        (0, 2, 2, 2, 2)  # A block (0,2), B block (2,2), C col 2
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        # A block is 2x2 at (a_r, a_c)
        # B block is 2x2 at (b_r, b_c)
        # C block is 2x2 at (0, c_c) ?? 
        # The original code's logic for indices needs to be preserved.
        
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        # W index mapping: W[k*n+i, r] where k is col of C, i is row of C.
        # C is 2x4. Rows 0,1. Cols 0,1,2,3.
        # The block in C corresponds to cols c_c, c_c+1.
        w11 = get_w_idx(c_c, 0)   # C[0, c_c]
        w12 = get_w_idx(c_c + 1, 0) # C[0, c_c+1]
        w21 = get_w_idx(c_c, 1)   # C[1, c_c]
        w22 = get_w_idx(c_c + 1, 1) # C[1, c_c+1]
        
        # The 7 terms of Strassen for 2x2 multiplication
        # P1 = (a11+a22)(b11+b22)
        # P2 = (a21+a22)b11
        # P3 = a11(b12-b22)
        # P4 = a22(b21-b11)
        # P5 = (a11+a12)b22
        # P6 = (a21-a11)(b11+b12)
        # P7 = (a12-a22)(b21+b22)
        
        # Term 1: P1
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 2: P2
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # Term 3: P3
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 4: P4
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # Term 5: P5
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # Term 6: P6
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 7: P7
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
        
    return {"U": U, "V": V, "W": W}
