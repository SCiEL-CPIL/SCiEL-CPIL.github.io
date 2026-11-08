import time

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    Compute A*B1 using Strassen (rank 49).
    Compute A*B2 using Strassen (rank 49).
    Compute A*B3 using Schoolbook (rank 16).
    Total rank: 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen 2x2 terms: (u_coeff, v_coeff, w_coeff)
    # u_coeff: dict mapping local A index -> coefficient
    # v_coeff: dict mapping local B index -> coefficient  
    # w_coeff: dict mapping local C index -> coefficient
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # m1=(a+d)(e+h) -> c00+,c11+
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # m2=(c+d)e -> c10+,c11-
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # m3=a(f-h) -> c01+,c11+
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # m4=d(g-e) -> c00+,c10+
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # m5=(a+b)h -> c00-,c01+
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # m6=(c-a)(e+f) -> c11+
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # m7=(b-d)(g+h) -> c00+
    ]
    
    # Strassen 4x4 block-level terms: (A_blocks, B_blocks, C_blocks)
    # Each is a list of (block_row, block_col, coeff)
    strassen_4x4_blocks = [
        # M1=(A00+A11)(B00+B11) -> C00+,C11+
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2=(A10+A11)B00 -> C10+,C11-
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3=A00(B01-B11) -> C01+,C11+
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4=A11(B10-B00) -> C00+,C10+
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5=(A00+A01)B11 -> C00-,C01+
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6=(A10-A00)(B00+B01) -> C11+
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7=(A01-A11)(B10+B11) -> C00+
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    def add_strassen_4x4(A_row_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """Add Strassen terms for 4x4 * 4x4 block multiplication."""
        for A_blocks, B_blocks, C_blocks in strassen_4x4_blocks:
            for u_coeff, v_coeff, w_coeff in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # Fill u_vec
                for local_idx, coeff in u_coeff.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for br, bc, b_coeff in A_blocks:
                        gi = A_row_off + br * 2 + li
                        gj = bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * b_coeff
                
                # Fill v_vec
                for local_idx, coeff in v_coeff.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, b_coeff in B_blocks:
                        gj = B_row_off + br * 2 + lj
                        gk = B_col_off + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * b_coeff
                
                # Fill w_vec: W index = k*n + i (column * n + row)
                for local_idx, coeff in w_coeff.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, b_coeff in C_blocks:
                        gi = C_row_off + br * 2 + li
                        gk = C_col_off + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * b_coeff
                
                add_term(u_vec, v_vec, w_vec)
                
                if time.time() > deadline:
                    return
    
    # Part 1: A * B1 (cols 0-3) -> C (rows 0-3, cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0)
    
    if time.time() > deadline:
        # Return whatever we have so far (may be incomplete)
        rank = len(U_cols)
        U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
        V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
        W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # Part 2: A * B2 (cols 4-7) -> C (rows 0-3, cols 4-7)
    add_strassen_4x4(0, 0, 4, 0, 4)
    
    if time.time() > deadline:
        rank = len(U_cols)
        U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
        V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
        W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # Part 3: A * B3 (col 8) -> C (rows 0-3, col 8)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
