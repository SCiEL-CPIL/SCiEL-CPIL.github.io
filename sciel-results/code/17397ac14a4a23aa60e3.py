import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 51
    rank = 51
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a 2x2x2 Strassen block
    def add_2x2_block(u_vec, v_vec, w_vec, a_blocks, b_blocks, c_blocks):
        # a_blocks, b_blocks, c_blocks are lists of (row_offset, col_offset) for 2x2 blocks
        # U is n x m flattened. Index i*m + k.
        # Block (br, bc) in A corresponds to i in [br*2, br*2+1], k in [bc*2, bc*2+1]
        for br, bc in a_blocks:
            for i_local in range(2):
                for k_local in range(2):
                    val = u_vec[i_local * 2 + k_local]
                    if val != 0:
                        i_global = br * 2 + i_local
                        k_global = bc * 2 + k_local
                        U[i_global * m + k_global, col_idx] += val
                        
        for br, bc in b_blocks:
            for k_local in range(2):
                for j_local in range(2):
                    val = v_vec[k_local * 2 + j_local]
                    if val != 0:
                        k_global = br * 2 + k_local
                        j_global = bc * 2 + j_local
                        V[k_global * p + j_global, col_idx] += val
                        
        for br, bc in c_blocks:
            for j_local in range(2):
                for i_local in range(2):
                    val = w_vec[j_local * 2 + i_local]
                    if val != 0:
                        j_global = br * 2 + j_local
                        i_global = bc * 2 + i_local
                        W[j_global * n + i_global, col_idx] += val

    # 7 Strassen terms for 2x2x2
    # U, V, W vectors are 4x1 (flattened 2x2)
    terms = [
        # M1: (A00+A11)(B00+B11)
        {'u': [1,0,0,1], 'v': [1,0,0,1], 'w': [1,0,0,1], 'a': [(0,0),(1,1)], 'b': [(0,0),(1,1)], 'c': [(0,0),(1,1)]},
        # M2: (A10+A11)B00
        {'u': [0,0,1,1], 'v': [1,0,0,0], 'w': [0,0,1,-1], 'a': [(1,0),(1,1)], 'b': [(0,0)], 'c': [(1,0),(1,1)]},
        # M3: A00(B10-B11)
        {'u': [1,0,0,0], 'v': [0,0,1,-1], 'w': [0,1,0,1], 'a': [(0,0)], 'b': [(1,0),(1,1)], 'c': [(0,1),(1,1)]},
        # M4: A11(B01-B00)
        {'u': [0,0,0,1], 'v': [-1,1,0,0], 'w': [1,0,1,0], 'a': [(1,1)], 'b': [(0,0),(0,1)], 'c': [(0,0),(1,0)]},
        # M5: (A00+A01)B11
        {'u': [1,1,0,0], 'v': [0,0,0,1], 'w': [-1,1,0,0], 'a': [(0,0),(0,1)], 'b': [(1,1)], 'c': [(0,0),(0,1)]},
        # M6: (A10-A00)(B00+B01)
        {'u': [-1,0,1,0], 'v': [1,1,0,0], 'w': [0,0,0,1], 'a': [(0,0),(1,0)], 'b': [(0,0),(0,1)], 'c': [(1,1)]},
        # M7: (A01-A11)(B10+B11)
        {'u': [0,1,0,-1], 'v': [0,0,1,1], 'w': [1,0,0,0], 'a': [(0,1),(1,1)], 'b': [(1,0),(1,1)], 'c': [(0,0)]},
    ]
    
    col_idx = 0
    for t in terms:
        u_vec = t['u']
        v_vec = t['v']
        w_vec = t['w']
        a_blocks = t['a']
        b_blocks = t['b']
        c_blocks = t['c']
        
        # Note: The add_2x2_block function needs to be modified to use col_idx
        # I will inline the logic or pass col_idx
        
        for br, bc in a_blocks:
            for i_local in range(2):
                for k_local in range(2):
                    val = u_vec[i_local * 2 + k_local]
                    if val != 0:
                        i_global = br * 2 + i_local
                        k_global = bc * 2 + k_local
                        U[i_global * m + k_global, col_idx] += val
                        
        for br, bc in b_blocks:
            for k_local in range(2):
                for j_local in range(2):
                    val = v_vec[k_local * 2 + j_local]
                    if val != 0:
                        k_global = br * 2 + k_local
                        j_global = bc * 2 + j_local
                        V[k_global * p + j_global, col_idx] += val
                        
        for br, bc in c_blocks:
            for j_local in range(2):
                for i_local in range(2):
                    val = w_vec[j_local * 2 + i_local]
                    if val != 0:
                        j_global = br * 2 + j_local
                        i_global = bc * 2 + i_local
                        W[j_global * n + i_global, col_idx] += val
                        
        col_idx += 1
        
    # Block 2: 4x4x1 tail (A1 b1)
    # U: 1 at i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i * m + k, col_idx] = 1
    # V: 1 at k in 0..3, j=4
    for k in range(4):
        V[k * p + 4, col_idx] = 1
    # W: 1 at i in 0..3, j=4
    for i in range(4):
        W[4 * n + i, col_idx] = 1
    col_idx += 1
    
    # Block 3: 4x1x5 tail (a b)
    # U: 1 at i in 0..3, k=4
    for i in range(4):
        U[i * m + 4, col_idx] = 1
    # V: 1 at k=4, j in 0..4
    for j in range(5):
        V[4 * p + j, col_idx] = 1
    # W: 1 at i in 0..3, j in 0..4
    for i in range(4):
        for j in range(5):
            W[j * n + i, col_idx] = 1
    col_idx += 1
    
    # Convert to lists of lists if required, or numpy arrays?
    # Prompt says: "factor matrices of shapes ... with integer/half-integer entries"
    # "construct() -> {'U', 'V', 'W'}"
    # The seed program returned lists of lists. I will return lists of lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
