import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition of rank 55 for multiplying a 2x3 matrix 
    by a 3x10 matrix using Strassen's algorithm on the 2x2 sub-problems.
    
    Strategy:
    - Split A (2x3) into A_L (2x2) and A_R (2x1).
    - Split B (3x10) into B_T (2x10) and B_B (1x10).
    - C = A_L @ B_T + A_R @ B_B.
    - A_R @ B_B is rank 20 (schoolbook for 2x1x10).
    - A_L @ B_T is sum of 5 (2x2x2) multiplications (splitting B_T's 10 cols into 5 pairs).
    - Each 2x2x2 multiplication uses Strassen (rank 7).
    - Total rank = 20 + 5*7 = 55.
    
    U shape: (n*m, rank) = (6, 55)
    V shape: (m*p, rank) = (30, 55)
    W shape: (p*n, rank) = (20, 55)
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Dimensions for sub-blocks
    # A_L is 2x2, A_R is 2x1
    # B_T is 2x10, B_B is 1x10
    
    # Rank components
    rank_schoolbook_part = 20  # For A_R (2x1) * B_B (1x10)
    num_strassen_blocks = 5    # 10 cols in B_T / 2 cols per block
    rank_strassen_part = 7 * num_strassen_blocks
    total_rank = rank_schoolbook_part + rank_strassen_part
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    col_offset = 0
    
    # Part 1: A_R (2x1) * B_B (1x10)
    # A_R corresponds to A[:, 2] (column 2 of A, since A is 2x3, indices 0,1 are left, 2 is right)
    # Wait, A is 2x3. Let's say A = [A_00 A_01 A_02; A_10 A_11 A_12].
    # A_L = [A_00 A_01; A_10 A_11] (cols 0,1)
    # A_R = [A_02; A_12] (col 2)
    # B is 3x10. B = [B_0; B_1; B_2] where B_0, B_1 are 1x10 rows? No, B is 3x10.
    # B_T = B[0:2, :] (rows 0,1) -> 2x10
    # B_B = B[2, :] (row 2) -> 1x10
    
    # Indices in flattened vectors:
    # A (2x3) flattened row-major: 
    # A_00: 0, A_01: 1, A_02: 2
    # A_10: 3, A_11: 4, A_12: 5
    # So A_L uses indices 0,1,3,4. A_R uses 2,5.
    
    # B (3x10) flattened row-major:
    # Row 0: 0..9
    # Row 1: 10..19
    # Row 2: 20..29
    # B_T uses rows 0,1 (indices 0..19).
    # B_B uses row 2 (indices 20..29).
    
    # C (2x10) flattened row-major:
    # Row 0: 0..9
    # Row 1: 10..19
    
    # W index convention: W[k*n + i, c] where C[i, k] is the element.
    # C is 2x10. i in 0..1, k in 0..9.
    # W row index: k*2 + i.
    
    # Part 1: A_R * B_B
    # A_R is 2x1. B_B is 1x10.
    # Product C_part1 = A_R * B_B.
    # C_part1[i, k] = A_R[i, 0] * B_B[0, k].
    # In tensor terms: T[i, j, k] for this part is non-zero only for j=2 (since A_R is col 2, B_B is row 2).
    # Wait, the tensor T[i, j, k] corresponds to A[i, j] * B[j, k].
    # For A_R, j=2. For B_B, j=2.
    # So we need T[i, 2, k] = A[i, 2] * B[2, k].
    
    # We decompose this into 20 rank-1 terms.
    # For each i in 0..1 and k in 0..9:
    #   u vector: 1 at index i*3 + 2 (A[i, 2])
    #   v vector: 1 at index 2*10 + k (B[2, k])
    #   w vector: 1 at index k*2 + i (C[i, k])
    
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U[u_idx][col_offset] = 1
            V[v_idx][col_offset] = 1
            W[w_idx][col_offset] = 1
            col_offset += 1
            
    # Part 2: A_L * B_T
    # A_L is 2x2. B_T is 2x10.
    # We split B_T into 5 blocks of 2x2.
    # B_T[:, 0:2], B_T[:, 2:4], ..., B_T[:, 8:10]
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will implement this for each of the 5 blocks.
    # For block b (0..4):
    #   B block columns are 2*b and 2*b+1.
    #   B indices in flattened B:
    #     B[0, 2b]   -> 0*10 + 2b
    #     B[0, 2b+1] -> 0*10 + 2b + 1
    #     B[1, 2b]   -> 1*10 + 2b
    #     B[1, 2b+1] -> 1*10 + 2b + 1
    
    #   A_L indices in flattened A:
    #     A[0, 0] -> 0
    #     A[0, 1] -> 1
    #     A[1, 0] -> 3
    #     A[1, 1] -> 4
    
    #   C block columns are 2*b and 2*b+1.
    #   C indices in flattened W (k*n+i):
    #     C[0, 2b]   -> 2b*2 + 0 = 4b
    #     C[0, 2b+1] -> (2b+1)*2 + 0 = 4b + 2
    #     C[1, 2b]   -> 2b*2 + 1 = 4b + 1
    #     C[1, 2b+1] -> (2b+1)*2 + 1 = 4b + 3
    
    # Strassen Tensors (u, v, w) for one block:
    # Note: u indices are relative to A_L (0,1,3,4). v indices relative to B_T block. w indices relative to C block.
    
    # We need to map local indices to global indices.
    
    def add_strassen_term(u_local, v_local, w_local, b):
        nonlocal col_offset
        # u_local: list of (index, val) in A_L space (0,1,3,4)
        # v_local: list of (index, val) in B_T block space (0,1,2,3) corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
        # w_local: list of (index, val) in C block space (0,1,2,3) corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
        
        for u_idx, u_val in u_local:
            # Map A_L index to global A index
            # A_L[0,0]=0, A_L[0,1]=1, A_L[1,0]=3, A_L[1,1]=4
            # These are already the global indices in A (0..5)
            global_u_idx = u_idx
            
            for v_idx, v_val in v_local:
                # Map B_T block index to global B index
                # Block b covers cols 2b, 2b+1.
                # v_idx 0 -> B[0, 2b] -> 0*10 + 2b
                # v_idx 1 -> B[0, 2b+1] -> 0*10 + 2b + 1
                # v_idx 2 -> B[1, 2b] -> 1*10 + 2b
                # v_idx 3 -> B[1, 2b+1] -> 1*10 + 2b + 1
                row, col_in_block = divmod(v_idx, 2)
                global_v_idx = row * 10 + (2 * b + col_in_block)
                
                for w_idx, w_val in w_local:
                    # Map C block index to global W index
                    # w_idx 0 -> C[0, 2b] -> 4b
                    # w_idx 1 -> C[0, 2b+1] -> 4b + 2
                    # w_idx 2 -> C[1, 2b] -> 4b + 1
                    # w_idx 3 -> C[1, 2b+1] -> 4b + 3
                    # Formula: w_idx = c_row * 2 + c_col_in_block? No.
                    # C block is 2x2.
                    # w_idx 0: (0,0) -> C[0, 2b]
                    # w_idx 1: (0,1) -> C[0, 2b+1]
                    # w_idx 2: (1,0) -> C[1, 2b]
                    # w_idx 3: (1,1) -> C[1, 2b+1]
                    c_row, c_col = divmod(w_idx, 2)
                    global_c_col = 2 * b + c_col
                    global_w_idx = global_c_col * 2 + c_row # k*n + i
                    
                    # Add to matrices
                    U[global_u_idx][col_offset] += u_val
                    V[global_v_idx][col_offset] += v_val
                    W[global_w_idx][col_offset] += w_val
        
        col_offset += 1

    # Define Strassen terms
    # u indices: 0=A00, 1=A01, 3=A10, 4=A11
    # v indices: 0=B00, 1=B01, 2=B10, 3=B11
    # w indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00 + A11)(B00 + B11)
    u1 = [(0, 1), (4, 1)]
    v1 = [(0, 1), (3, 1)]
    w1 = [(0, 1), (3, -1)] # C00 += M1, C11 -= M1 ? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the contribution of each M to the C matrix.
    # M1 contributes to C00 (+1) and C11 (+1)
    # M2 contributes to C10 (+1) and C11 (-1)
    # M3 contributes to C01 (+1) and C11 (+1)
    # M4 contributes to C00 (+1) and C10 (+1)
    # M5 contributes to C00 (-1) and C01 (+1)
    # M6 contributes to C11 (+1)
    # M7 contributes to C00 (+1)
    
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    terms = [
        # (u, v, w)
        ([(0, 1), (4, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)]),        # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),        # M3
        ([(4, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),        # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),        # M5
        ([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),        # M6
        ([(1, 1), (4, -1)], [(2, 1), (3, 1)], [(0, 1)])         # M7
    ]
    
    for b in range(5):
        for u_t, v_t, w_t in terms:
            add_strassen_term(u_t, v_t, w_t, b)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
