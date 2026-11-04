"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 10
    total_rank = 55
    
    # Initialize factor matrices with zeros
    U_mat = [[0] * total_rank for _ in range(n * m)]
    V_mat = [[0] * total_rank for _ in range(m * p)]
    W_mat = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0

    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U part (rows of A): a=0, b=1, c=3, d=4 (for 2x2 block)
    # V part (rows of B): e=0, f=1, g=10, h=11 (relative to 2x10 block start)
    # W part (cols of C): C00=0, C10=1, C01=2, C11=3 (relative to 2x2 block start)
    
    strassen_defs = [
        # (U_rows: [(row_offset, coeff)], V_rows: [(row_offset, coeff)], W_rows: [(row_offset, coeff)])
        # m1: (a+d)(e+h) -> C00, C11
        {
            'U': [(0, 1), (4, 1)],
            'V': [(0, 1), (11, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # m2: (c+d)e -> C10, -C11
        {
            'U': [(3, 1), (4, 1)],
            'V': [(0, 1)],
            'W': [(1, 1), (3, -1)]
        },
        # m3: a(f-h) -> C01, C11
        {
            'U': [(0, 1)],
            'V': [(1, 1), (11, -1)],
            'W': [(2, 1), (3, 1)]
        },
        # m4: d(g-e) -> C00, C10
        {
            'U': [(4, 1)],
            'V': [(10, 1), (0, -1)],
            'W': [(0, 1), (1, 1)]
        },
        # m5: (a+b)h -> -C00, C01
        {
            'U': [(0, 1), (1, 1)],
            'V': [(11, 1)],
            'W': [(0, -1), (2, 1)]
        },
        # m6: (c-a)(e+f) -> C11
        {
            'U': [(3, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # m7: (b-d)(g+h) -> C00
        {
            'U': [(1, 1), (4, -1)],
            'V': [(10, 1), (11, 1)],
            'W': [(0, 1)]
        }
    ]

    # Process 5 disjoint 2x2 blocks of B (columns 0-9)
    for block_idx in range(5):
        # Column offsets in B
        col_start = block_idx * 2
        # V rows offset: B is 3x10. Top 2 rows are indices 0..9.
        # For block_idx, cols are col_start, col_start+1
        # V row indices:
        # e (row 0, col c0) -> 0 + c0
        # f (row 0, col c1) -> 0 + c1
        # g (row 1, col c0) -> 10 + c0
        # h (row 1, col c1) -> 10 + c1
        
        # W columns offset: C is 2x10.
        # For block_idx, cols are col_start, col_start+1
        # W row indices (k*n + i):
        # C[0, c0] -> k=c0, i=0 -> c0*2 + 0
        # C[1, c0] -> k=c0, i=1 -> c0*2 + 1
        # C[0, c1] -> k=c1, i=0 -> c1*2 + 0
        # C[1, c1] -> k=c1, i=1 -> c1*2 + 1

        for s in range(7):
            defn = strassen_defs[s]
            
            # U rows are global for the 2x2 A block (rows 0,1,3,4 of A)
            # A is 2x3. Rows 0,1 are the full row.
            # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=3, d=A[1,1]=4 (row-major 2x2 part of 2x3)
            for u_row, coeff in defn['U']:
                U_mat[u_row][current_rank] += coeff
            
            # V rows are relative to the block
            for v_rel_row, coeff in defn['V']:
                # Map relative row to absolute row in V
                # v_rel_row 0 -> col c0, row 0
                # v_rel_row 1 -> col c1, row 0
                # v_rel_row 10 -> col c0, row 1
                # v_rel_row 11 -> col c1, row 1
                if v_rel_row < 10:
                    col_offset = v_rel_row
                    row_base = 0
                else:
                    col_offset = v_rel_row - 10
                    row_base = 10
                
                abs_v_row = row_base + col_start + col_offset
                V_mat[abs_v_row][current_rank] += coeff

            # W rows are relative to the block
            for w_rel_row, coeff in defn['W']:
                # Map relative row to absolute row in W
                # w_rel_row 0 -> C[0, c0]
                # w_rel_row 1 -> C[1, c0]
                # w_rel_row 2 -> C[0, c1]
                # w_rel_row 3 -> C[1, c1]
                
                # c0 = col_start, c1 = col_start + 1
                if w_rel_row < 2:
                    col_offset = 0 # c0
                    row_in_c = w_rel_row # i=0 or 1
                else:
                    col_offset = 1 # c1
                    row_in_c = w_rel_row - 2 # i=0 or 1
                
                k = col_start + col_offset
                i = row_in_c
                abs_w_row = k * n + i
                W_mat[abs_w_row][current_rank] += coeff

            current_rank += 1

    # Process the remaining term: A_R (2x1) * B_B (1x10)
    # A_R is column 2 of A. Indices in U:
    # A[0,2] -> index 2
    # A[1,2] -> index 5
    # B_B is row 2 of B. Indices in V:
    # B[2,0..9] -> indices 20..29
    
    # We need to compute C[i, k] += A[i, 2] * B[2, k]
    # This is a simple outer product sum, requiring 2*10 = 20 terms.
    
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][current_rank] = 1
            V_mat[v_idx][current_rank] = 1
            W_mat[w_idx][current_rank] = 1
            current_rank += 1

    # Validation
    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)
    
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if not is_valid:
        # Fallback to trivial rank-60
        rank60 = 60
        U2 = [[0] * rank60 for _ in range(n * m)]
        V2 = [[0] * rank60 for _ in range(m * p)]
        W2 = [[0] * rank60 for _ in range(p * n)]
        rr = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U2[ii * m + jj][rr] = 1
                    V2[jj * p + kk][rr] = 1
                    W2[kk * n + ii][rr] = 1
                    rr += 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
