import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = time.time() + 140  # 140s deadline, 20s safety margin from 160s limit

def construct():
    n, m, p = 3, 4, 6
    
    # Rank 66 decomposition
    # Strategy:
    # 1. Top row (1x4) * (4x6) -> Schoolbook (24 mults)
    # 2. Bottom rows (2x4) * (4x6) -> Split A into two 2x2 blocks (Left, Right)
    #    Split B into three 2x2 column blocks (Col1, Col2, Col3)
    #    For each A block and B block pair, use Strassen (7 mults)
    #    Total bottom: 2 (A blocks) * 3 (B blocks) * 7 = 42 mults
    # Total: 24 + 42 = 66
    
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
            return
        for idx, val in u_specs:
            if val != 0:
                U[idx, curr_rank] += val
        for idx, val in v_specs:
            if val != 0:
                V[idx, curr_rank] += val
        for idx, val in w_specs:
            if val != 0:
                W[idx, curr_rank] += val
        curr_rank += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen Coefficients for 2x2x2
    # Local A: [[a0, a1], [a2, a3]]
    # Local B: [[b0, b1], [b2, b3]]
    # Local C: [[c0, c1], [c2, c3]]
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    
    # C reconstruction:
    # c0 = M1 - M5 + M7
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M6
    
    s_defs = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7
    ]

    # 1. Top Row: Schoolbook
    # A[0, j] * B[j, k] -> C[k, 0]
    for j in range(m):
        for k in range(p):
            u_spec = [(u_idx(0, j), 1)]
            v_spec = [(v_idx(j, k), 1)]
            w_spec = [(w_idx(k, 0), 1)]
            add_mults(u_spec, v_spec, w_spec)

    # 2. Bottom Rows: Strassen
    # A[1:3, :] is 2x4. Split into Left (cols 0,1) and Right (cols 2,3)
    # B[:, :] is 4x6. Split into 3 blocks of 2 columns: (0,1), (2,3), (4,5)
    
    a_row_off = 1
    a_col_offs = [0, 2] # Left, Right
    b_row_offs = [0, 2] # Top, Bottom
    b_col_offs = [0, 2, 4] # 3 blocks
    
    for a_col_off in a_col_offs:
        for b_row_off in b_row_offs:
            for b_col_off in b_col_offs:
                # Define local indices for this 2x2x2x2 block
                # A local: rows 1,2; cols a_col_off, a_col_off+1
                # B local: rows b_row_off, b_row_off+1; cols b_col_off, b_col_off+1
                # C local: rows 1,2; cols b_col_off, b_col_off+1
                
                def get_a_idx(loc):
                    r, c = divmod(loc, 2)
                    return u_idx(a_row_off + r, a_col_off + c)
                
                def get_b_idx(loc):
                    r, c = divmod(loc, 2)
                    return v_idx(b_row_off + r, b_col_off + c)
                
                def get_c_idx(loc):
                    r, c = divmod(loc, 2)
                    return w_idx(b_col_off + c, a_row_off + r)

                for a_c, b_c, c_c in s_defs:
                    u_spec = [(get_a_idx(i), val) for i, val in enumerate(a_c) if val != 0]
                    v_spec = [(get_b_idx(i), val) for i, val in enumerate(b_c) if val != 0]
                    w_spec = [(get_c_idx(i), val) for i, val in enumerate(c_c) if val != 0]
                    add_mults(u_spec, v_spec, w_spec)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
