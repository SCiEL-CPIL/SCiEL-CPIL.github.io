import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_2x2(i_off, j_off, k_off):
        # Adds 7 columns for a 2x2x2 block
        # Factor 0: M1 = (A00 + A11)(B00 + B11) -> C00, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 1: M2 = (A10 + A11)B00 -> C10, C11(-1)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+0)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        w_vec[(k_off+0)*5 + (i_off+1)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = -1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 2: M3 = A00(B01 - B11) -> C01, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = -1
        w_vec[(k_off+1)*5 + (i_off+0)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 3: M4 = A11(B10 - B00) -> C00, C10
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = -1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        w_vec[(k_off+0)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 4: M5 = (A00 + A01)B11 -> C00(-1), C01
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        u_vec[(i_off+0)*5 + (j_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = -1
        w_vec[(k_off+1)*5 + (i_off+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 5: M6 = (A10 - A00)(B00 + B01) -> C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+0)] = 1
        u_vec[(i_off+0)*5 + (j_off+0)] = -1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+1)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 6: M7 = (A01 - A11)(B10 + B11) -> C00
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+1)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = -1
        v_vec[(j_off+1)*5 + (k_off+0)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        add_col(u_vec, v_vec, w_vec)

    def add_strassen_4x4(i_off, j_off, k_off):
        # Block Strassen for 4x4x4 (Rank 49)
        # Helper to add a rank-1 tensor for the inner 2x2 Strassen
        def add_inner_term(u_indices, v_indices, w_indices):
            u_vec = [0]*25
            v_vec = [0]*25
            w_vec = [0]*25
            for idx, val in u_indices.items():
                u_vec[idx] = val
            for idx, val in v_indices.items():
                v_vec[idx] = val
            for idx, val in w_indices.items():
                w_vec[idx] = val
            add_col(u_vec, v_vec, w_vec)

        # Define block offsets for the 4x4 block
        # A blocks:
        A00 = [(i_off + r) * 5 + (j_off + c) for r in range(2) for c in range(2)]
        A01 = [(i_off + r) * 5 + (j_off + 2 + c) for r in range(2) for c in range(2)]
        A10 = [(i_off + 2 + r) * 5 + (j_off + c) for r in range(2) for c in range(2)]
        A11 = [(i_off + 2 + r) * 5 + (j_off + 2 + c) for r in range(2) for c in range(2)]
        
        # B blocks
        B00 = [(j_off + r) * 5 + (k_off + c) for r in range(2) for c in range(2)]
        B01 = [(j_off + r) * 5 + (k_off + 2 + c) for r in range(2) for c in range(2)]
        B10 = [(j_off + 2 + r) * 5 + (k_off + c) for r in range(2) for c in range(2)]
        B11 = [(j_off + 2 + r) * 5 + (k_off + 2 + c) for r in range(2) for c in range(2)]
        
        # C blocks
        C00 = [(k_off + r) * 5 + (i_off + c) for r in range(2) for c in range(2)]
        C01 = [(k_off + r) * 5 + (i_off + 2 + c) for r in range(2) for c in range(2)]
        C10 = [(k_off + 2 + r) * 5 + (i_off + c) for r in range(2) for c in range(2)]
        C11 = [(k_off + 2 + r) * 5 + (i_off + 2 + c) for r in range(2) for c in range(2)]

        # Helper to create a dict from a list of indices and a value
        def make_dict(indices, val=1):
            d = {}
            for idx in indices:
                d[idx] = d.get(idx, 0) + val
            return d

        # Outer Strassen Terms
        # M1 = (A00 + A11)(B00 + B11)
        # Inner Strassen for M1:
        # X = A00+A11, Y = B00+B11
        # N1 = (X00+X11)(Y00+Y11) -> U: X00, X11; V: Y00, Y11; W: C00, C11
        # X00 is A00[0,0], X11 is A11[1,1]? No.
        # X is a 2x2 matrix. X[0,0] is A00[0,0]. X[1,1] is A11[1,1].
        # Wait, A00 is a 2x2 block. A00[0,0] is the top-left element of A00.
        # So X[0,0] = A00[0,0]. X[0,1] = A00[0,1]. X[1,0] = A00[1,0]. X[1,1] = A00[1,1].
        # And X[0,0] also gets A11[0,0]? No.
        # X = A00 + A11.
        # So X[0,0] = A00[0,0] + A11[0,0].
        # X[0,1] = A00[0,1] + A11[0,1].
        # X[1,0] = A00[1,0] + A11[1,0].
        # X[1,1] = A00[1,1] + A11[1,1].
        
        # Let's map the inner indices 0..3 to global indices.
        # Inner index 0: (0,0) -> A00[0,0], A11[0,0]
        # Inner index 1: (0,1) -> A00[0,1], A11[0,1]
        # Inner index 2: (1,0) -> A00[1,0], A11[1,0]
        # Inner index 3: (1,1) -> A00[1,1], A11[1,1]
        
        def get_block_indices(block_list, r, c):
            return block_list[r*2+c]

        # M1: X = A00+A11, Y = B00+B11
        # X indices:
        X0 = [get_block_indices(A00,0,0), get_block_indices(A11,0,0)]
        X1 = [get_block_indices(A00,0,1), get_block_indices(A11,0,1)]
        X2 = [get_block_indices(A00,1,0), get_block_indices(A11,1,0)]
        X3 = [get_block_indices(A00,1,1), get_block_indices(A11,1,1)]
        # Y indices:
        Y0 = [get_block_indices(B00,0,0), get_block_indices(B11,0,0)]
        Y1 = [get_block_indices(B00,0,1), get_block_indices(B11,0,1)]
        Y2 = [get_block_indices(B00,1,0), get_block_indices(B11,1,0)]
        Y3 = [get_block_indices(B00,1,1), get_block_indices(B11,1,1)]
        # Z indices (C00, C11):
        # Z0 -> C00[0,0], Z1 -> C00[0,1], Z2 -> C00[1,0], Z3 -> C00[1,1]
        # Also Z0 contributes to C11? No.
        # The W part of the rank-1 tensor for N1 in M1's decomposition:
        # N1 contributes to Z[0,0] and Z[1,1] of the 2x2 result M1.
        # M1 contributes to C00 and C11.
        # So N1 contributes to C00[0,0] and C11[0,0]? No.
        # M1 is a 2x2 matrix. M1[0,0] adds to C00[0,0]. M1[1,1] adds to C11[1,1]?
        # No.
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So M1[0,0] adds to C00[0,0] and C11[0,0].
        # M1[0,1] adds to C00[0,1] and C11[0,1].
        # M1[1,0] adds to C00[1,0] and C11[1,0].
        # M1[1,1] adds to C00[1,1] and C11[1,1].
        
        # So for N1 (which is M1's inner term 1):
        # N1 contributes to M1[0,0] and M1[1,1].
        # So W has 1s at C00[0,0], C11[0,0], C00[1,1], C11[1,1].
        
        # This is getting complicated to map.
        # Let's use a simpler approach: Just implement the 49 vectors explicitly or use a library.
        # Since I can't use a library, I will write the 49 vectors.
        # But that's a lot of code.
        
        # Alternative: Use the 2+2+1 partition with rank 117.
        # It is already implemented and working.
        # I will stick to 117 to avoid bugs.
        pass

    # Since the current program is already at 117, and I cannot easily find a better one,
    # I will return the current program.
    
    # Wait, I can try to use the 4x4 Strassen (rank 49) by implementing it properly.
    # Let's do it.
    
    def add_4x4_strassen_impl(i_off, j_off, k_off):
        # 7 terms
        # M1 = (A00 + A11)(B00 + B11)
        # We need to add the rank-7 decomposition of this product.
        # The product is of two 2x2 matrices.
        # We can call add_strassen_2x2 if we could map the blocks.
        # Instead, we will manually add the 7 factors for each term.
        
        # Term 1: (A00 + A11)(B00 + B11)
        # U: A00, A11. V: B00, B11. W: C00, C11.
        # This is equivalent to a 2x2 Strassen block where the "scalars" are 2x2 blocks.
        # We can just call add_strassen_2x2 for the 4 quadrants? No.
        
        # I will use the Tiling approach (rank 56) for 4x4.
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    add_strassen_2x2(i_off + di*2, j_off + dj*2, k_off + dk*2)

    # Partition 4+1
    # 4x4x4: Tiling Strassen (rank 56)
    add_4x4_strassen_impl(0, 0, 0)
    
    # 4x4x1: Schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            u_vec[i*5 + j] = 1
            v_vec[j*5 + 4] = 1
            w_vec[4*5 + i] = 1
            add_col(u_vec, v_vec, w_vec)
            
    # 4x1x4: Schoolbook
    for i in range(4):
        for k in range(4):
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            u_vec[i*5 + 4] = 1
            v_vec[4*5 + k] = 1
            w_vec[k*5 + i] = 1
            add_col(u_vec, v_vec, w_vec)
            
    # 1x4x4: Schoolbook
    for j in range(4):
        for k in range(4):
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            u_vec[4*5 + j] = 1
            v_vec[j*5 + k] = 1
            w_vec[k*5 + 4] = 1
            add_col(u_vec, v_vec, w_vec)
            
    # 4x1x1: Schoolbook
    for i in range(4):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[i*5 + 4] = 1
        v_vec[4*5 + 4] = 1
        w_vec[4*5 + i] = 1
        add_col(u_vec, v_vec, w_vec)
        
    # 1x4x1: Schoolbook
    for j in range(4):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[4*5 + j] = 1
        v_vec[j*5 + 4] = 1
        w_vec[4*5 + 4] = 1
        add_col(u_vec, v_vec, w_vec)
        
    # 1x1x4: Schoolbook
    for k in range(4):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[4*5 + 4] = 1
        v_vec[4*5 + k] = 1
        w_vec[k*5 + 4] = 1
        add_col(u_vec, v_vec, w_vec)
        
    # 1x1x1: Schoolbook
    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
    u_vec[4*5 + 4] = 1
    v_vec[4*5 + 4] = 1
    w_vec[4*5 + 4] = 1
    add_col(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
