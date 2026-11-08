import numpy as np
import time

def construct():
    """
    Rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7 for 2x2x2) to decompose the
    multiplication into blocks, then apply schoolbook for the remaining row.
    
    Block decomposition:
    - A is 3x4, B is 4x6, C is 3x6
    - Split A into A1 (row 0, 1x4) and A2 (rows 1-2, 2x4)
    - Split B into B1 (cols 0-1, 4x2), B2 (cols 2-3, 4x2), B3 (cols 4-5, 4x2)
    - C1 = A1*B1 (1x2), C2 = A1*B2 (1x2), C3 = A1*B3 (1x2)
    - C4 = A2*B1 (2x2), C5 = A2*B2 (2x2), C6 = A2*B3 (2x2)
    
    For A1*Bk (1x4 * 4x2): use schoolbook, rank = 4*2 = 8 each, total 24
    For A2*Bk (2x4 * 4x2): split into (2x2)*(2x2) blocks, use Strassen, rank = 7 each
    Total Strassen: 3 blocks * 2 sub-blocks * 7 = 42
    Total rank: 24 + 42 = 66
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = 3, 4, 6
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i
    
    # Strassen decomposition for 2x2x2
    # Each entry is (u_vec, v_vec, w_vec) of length 4
    # u_vec[r*2+c] corresponds to A[r,c], v_vec[r*2+c] to B[r,c], w_vec[r*2+c] to C[r,c]
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    rank = 66
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    
    def add_term(u_map, v_map, w_map):
        nonlocal col
        if col >= rank:
            return False
        for idx, val in u_map:
            U[idx, col] = val
        for idx, val in v_map:
            V[idx, col] = val
        for idx, val in w_map:
            W[idx, col] = val
        col += 1
        return True
    
    # Part 1: Row 0 of A times each 4x2 block of B (schoolbook)
    for b_block in range(3):  # 3 blocks of width 2
        if time.time() > deadline:
            break
        b_col_start = b_block * 2
        for j in range(4):    # 4 rows of B
            for k in range(2): # 2 cols in this block
                u = [(ui(0, j), 1.0)]
                v = [(vi(j, b_col_start + k), 1.0)]
                w = [(wi(b_col_start + k, 0), 1.0)]
                add_term(u, v, w)
    
    # Part 2: Rows 1-2 of A times each 4x2 block of B (Strassen)
    for b_block in range(3):
        if time.time() > deadline:
            break
        b_col_start = b_block * 2
        
        # Left half: A[1:3, 0:2] * B[0:2, b_col_start:b_col_start+2]
        for u_vec, v_vec, w_vec in S:
            u_map = []
            v_map = []
            w_map = []
            for idx, val in enumerate(u_vec):
                if val != 0:
                    r, c = divmod(idx, 2)
                    u_map.append((ui(1+r, c), float(val)))
            for idx, val in enumerate(v_vec):
                if val != 0:
                    r, c = divmod(idx, 2)
                    v_map.append((vi(r, b_col_start+c), float(val)))
            for idx, val in enumerate(w_vec):
                if val != 0:
                    r, c = divmod(idx, 2)
                    w_map.append((wi(b_col_start+c, 1+r), float(val)))
            add_term(u_map, v_map, w_map)
        
        # Right half: A[1:3, 2:4] * B[2:4, b_col_start:b_col_start+2]
        for u_vec, v_vec, w_vec in S:
            u_map = []
            v_map = []
            w_map = []
            for idx, val in enumerate(u_vec):
                if val != 0:
                    r, c = divmod(idx, 2)
                    u_map.append((ui(1+r, 2+c), float(val)))
            for idx, val in enumerate(v_vec):
                if val != 0:
                    r, c = divmod(idx, 2)
                    v_map.append((vi(2+r, b_col_start+c), float(val)))
            for idx, val in enumerate(w_vec):
                if val != 0:
                    r, c = divmod(idx, 2)
                    w_map.append((wi(b_col_start+c, 1+r), float(val)))
            add_term(u_map, v_map, w_map)
    
    # Trim any zero columns
    mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
