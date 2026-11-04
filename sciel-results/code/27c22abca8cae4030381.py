import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    deadline = start_time + 140

    # 2-level decomposition:
    # Outer: 2x2 blocks of 3x3 submatrices, using Strassen (rank 7)
    # Inner: 3x3 naive multiplication (rank 27)
    # Total rank: 7 * 27 = 189
    
    block_size = 3
    num_blocks = 2
    total_rank = 189

    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)

    # Strassen for 2x2 block multiplication
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (block_row, block_col, coeff) for A blocks
    # v_coeffs: list of (block_row, block_col, coeff) for B blocks
    # w_coeffs: list of (block_row, block_col, coeff) for C blocks
    strassen_2x2 = [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A21+A22)B11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A11(B21-B11)
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1)]),
        # M4 = A22(B12-B22)
        ([(1,1,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]),
        # M5 = (A11+A12)B22
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1)]),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(0,1,1)]),
    ]

    # Naive 3x3 multiplication: 27 terms
    # Each term: (a_row, a_col, b_row, b_col, c_row, c_col)
    naive_3x3 = []
    for ar in range(3):
        for ac in range(3):
            for br in range(3):
                for bc in range(3):
                    # A[ar,ac] * B[br,bc] contributes to C[ar,bc]
                    naive_3x3.append((ar, ac, br, bc, ar, bc))

    col = 0
    bs = block_size
    dim = 6

    for u_list, v_list, w_list in strassen_2x2:
        for (u_br, u_bc, u_coeff) in u_list:
            for (v_br, v_bc, v_coeff) in v_list:
                # For each combination of A block (u_br, u_bc) and B block (v_br, v_bc),
                # we need to do naive 3x3 multiplication
                for (ar, ac, br, bc, cr, cc) in naive_3x3:
                    # Global indices
                    # A element: row = u_br*3 + ar, col = u_bc*3 + ac
                    a_row = u_br * bs + ar
                    a_col = u_bc * bs + ac
                    # B element: row = v_br*3 + br, col = v_bc*3 + bc
                    b_row = v_br * bs + br
                    b_col = v_bc * bs + bc
                    
                    U[a_row * dim + a_col, col] += u_coeff
                    V[b_row * dim + b_col, col] += v_coeff
                    
                    # C element: for each w coefficient
                    for (w_br, w_bc, w_coeff) in w_list:
                        c_row = w_br * bs + cr
                        c_col = w_bc * bs + cc
                        # Trace convention: W[k*n+i, r] where i is row of C, k is col of C
                        # So W index = c_col * n + c_row
                        W[c_col * dim + c_row, col] += w_coeff
                    
                    col += 1
                    
                    if col % 100 == 0 and time.time() > deadline:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
