"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 via Strassen on 2x2 blocks.
Splits the 4-dimension into two 2-dimensions, giving 2x2 @ 2x6 subproblems.
Each 2x2 @ 2x6 is split into three 2x2 @ 2x2 blocks, each using Strassen (rank 7).
Total rank = 2 * 3 * 7 = 42.
"""

def construct():
    n, m, p = 2, 4, 6

    # Strassen 2x2 @ 2x2: 7 rank-1 terms
    # Each term is a tuple (U_sparse, V_sparse, W_sparse)
    # U_sparse: list of (local_idx, coeff)
    # local_idx: 0=a11, 1=a12, 2=a21, 3=a22
    # V_sparse: list of (local_idx, coeff)
    # local_idx: 0=b11, 1=b12, 2=b21, 3=b22
    # W_sparse: list of (local_idx, coeff)
    # local_idx: 0=c11, 1=c12, 2=c21, 3=c22
    
    strassen_terms = [
        # M1 = (a11+a22)(b11+b22) -> c11, c22
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a21+a22)b11 -> c21, -c22
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a11(b12-b22) -> c12, c22
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a22(b21-b11) -> c11, c21
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a11+a12)b22 -> -c11, c12
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a21-a11)(b11+b12) -> c22
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a12-a22)(b21+b22) -> c11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42

    current_rank = 0
    for side in range(2):  # 0: left A (cols 0-1), top B (rows 0-1); 1: right A (cols 2-3), bottom B (rows 2-3)
        u_col_off = side * 2
        v_row_off = side * 2
        
        for col_block in range(3):  # 3 blocks of 2 columns
            v_col_off = col_block * 2
            
            for u_sparse, v_sparse, w_sparse in strassen_terms:
                # Process U
                for local_a_idx, coeff in u_sparse:
                    # A is 2x4. A[row, col]
                    # local_a_idx: 0=a11 (0,0), 1=a12 (0,1), 2=a21 (1,0), 3=a22 (1,1)
                    a_row = local_a_idx // 2
                    a_col_local = local_a_idx % 2
                    a_col_global = u_col_off + a_col_local
                    u_row_global = a_row * m + a_col_global
                    U[u_row_global][current_rank] += coeff

                # Process V
                for local_b_idx, coeff in v_sparse:
                    # B is 4x6. B[row, col]
                    # local_b_idx: 0=b11 (0,0), 1=b12 (0,1), 2=b21 (1,0), 3=b22 (1,1) relative to the 2x2 block
                    # The 2x2 block of B is at rows [v_row_off, v_row_off+1] and cols [v_col_off, v_col_off+1]
                    b_row_local = local_b_idx // 2
                    b_col_local = local_b_idx % 2
                    b_row_global = v_row_off + b_row_local
                    b_col_global = v_col_off + b_col_local
                    v_row_global = b_row_global * p + b_col_global
                    V[v_row_global][current_rank] += coeff

                # Process W
                for local_c_idx, coeff in w_sparse:
                    # C is 2x6. C[row, col]
                    # local_c_idx: 0=c11 (0,0), 1=c12 (0,1), 2=c21 (1,0), 3=c22 (1,1) relative to the 2x2 block
                    # The 2x2 block of C is at rows [0, 1] and cols [v_col_off, v_col_off+1]
                    # Note: C rows are always 0,1 because A has 2 rows.
                    c_row_local = local_c_idx // 2
                    c_col_local = local_c_idx % 2
                    c_row_global = c_row_local
                    c_col_global = v_col_off + c_col_local
                    
                    # Trace convention: W[k*n+i, r] where k is col of C, i is row of C
                    # So W index = c_col_global * n + c_row_global
                    w_row_global = c_col_global * n + c_row_global
                    W[w_row_global][current_rank] += coeff

                current_rank += 1

    return {"U": U, "V": V, "W": W}
