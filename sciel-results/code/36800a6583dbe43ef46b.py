import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2x2
    # P1: (A11+A22)(B11+B22) -> C11+=1, C22+=1
    # P2: (A21+A22)B11 -> C21+=1, C22-=1
    # P3: A11(B12-B22) -> C12+=1, C22+=1
    # P4: A22(B21-B11) -> C11+=1, C21+=1
    # P5: (A11+A12)B22 -> C11-=1, C12+=1
    # P6: (A21-A11)(B11+B12) -> C22+=1
    # P7: (A12-A22)(B21+B22) -> C11+=1

    def add_term(a_coeffs, b_coeffs, c_coeffs, idx):
        for (ai, val) in a_coeffs:
            U[ai, idx] += val
        for (bi, val) in b_coeffs:
            V[bi, idx] += val
        for (ci, val) in c_coeffs:
            W[ci, idx] += val

    def a_idx(i, j, col_offset):
        return i * m + (j + col_offset)

    def b_idx(i, j, row_offset, col_offset):
        return (i + row_offset) * p + (j + col_offset)

    def c_idx(i, j, col_offset):
        return (j + col_offset) * n + i

    col = 0
    # Group 1: A[:, 0:2] * B[0:2, :]
    for block_col in [0, 1, 2]:
        # P1
        add_term([(a_idx(0,0,0),1), (a_idx(1,1,0),1)], 
                 [(b_idx(0,0,0,block_col),1), (b_idx(1,1,0,block_col),1)],
                 [(c_idx(0,0,block_col),1), (c_idx(1,1,block_col),1)], col)
        col += 1
        # P2
        add_term([(a_idx(1,0,0),1), (a_idx(1,1,0),1)], 
                 [(b_idx(0,0,0,block_col),1)],
                 [(c_idx(1,0,block_col),1), (c_idx(1,1,block_col),-1)], col)
        col += 1
        # P3
        add_term([(a_idx(0,0,0),1)], 
                 [(b_idx(0,1,0,block_col),1), (b_idx(1,1,0,block_col),-1)],
                 [(c_idx(0,1,block_col),1), (c_idx(1,1,block_col),1)], col)
        col += 1
        # P4
        add_term([(a_idx(1,1,0),1)], 
                 [(b_idx(1,0,0,block_col),1), (b_idx(0,0,0,block_col),-1)],
                 [(c_idx(0,0,block_col),1), (c_idx(1,0,block_col),1)], col)
        col += 1
        # P5
        add_term([(a_idx(0,0,0),1), (a_idx(0,1,0),1)], 
                 [(b_idx(1,1,0,block_col),1)],
                 [(c_idx(0,0,block_col),-1), (c_idx(0,1,block_col),1)], col)
        col += 1
        # P6
        add_term([(a_idx(1,0,0),1), (a_idx(0,0,0),-1)], 
                 [(b_idx(0,0,0,block_col),1), (b_idx(0,1,0,block_col),1)],
                 [(c_idx(1,1,block_col),1)], col)
        col += 1
        # P7
        add_term([(a_idx(0,1,0),1), (a_idx(1,1,0),-1)], 
                 [(b_idx(1,0,0,block_col),1), (b_idx(1,1,0,block_col),1)],
                 [(c_idx(0,0,block_col),1)], col)
        col += 1

    # Group 2: A[:, 2:4] * B[2:4, :]
    for block_col in [0, 1, 2]:
        # P1
        add_term([(a_idx(0,2,0),1), (a_idx(1,3,0),1)], 
                 [(b_idx(0,0,2,block_col),1), (b_idx(1,1,2,block_col),1)],
                 [(c_idx(0,0,block_col),1), (c_idx(1,1,block_col),1)], col)
        col += 1
        # P2
        add_term([(a_idx(1,2,0),1), (a_idx(1,3,0),1)], 
                 [(b_idx(0,0,2,block_col),1)],
                 [(c_idx(1,0,block_col),1), (c_idx(1,1,block_col),-1)], col)
        col += 1
        # P3
        add_term([(a_idx(0,2,0),1)], 
                 [(b_idx(0,1,2,block_col),1), (b_idx(1,1,2,block_col),-1)],
                 [(c_idx(0,1,block_col),1), (c_idx(1,1,block_col),1)], col)
        col += 1
        # P4
        add_term([(a_idx(1,3,0),1)], 
                 [(b_idx(1,0,2,block_col),1), (b_idx(0,0,2,block_col),-1)],
                 [(c_idx(0,0,block_col),1), (c_idx(1,0,block_col),1)], col)
        col += 1
        # P5
        add_term([(a_idx(0,2,0),1), (a_idx(0,3,0),1)], 
                 [(b_idx(1,1,2,block_col),1)],
                 [(c_idx(0,0,block_col),-1), (c_idx(0,1,block_col),1)], col)
        col += 1
        # P6
        add_term([(a_idx(1,2,0),1), (a_idx(0,2,0),-1)], 
                 [(b_idx(0,0,2,block_col),1), (b_idx(0,1,2,block_col),1)],
                 [(c_idx(1,1,block_col),1)], col)
        col += 1
        # P7
        add_term([(a_idx(0,3,0),1), (a_idx(1,3,0),-1)], 
                 [(b_idx(1,0,2,block_col),1), (b_idx(1,1,2,block_col),1)],
                 [(c_idx(0,0,block_col),1)], col)
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
