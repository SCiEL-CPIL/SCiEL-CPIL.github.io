import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rows for U, V, W
    u_rows = n * m  # 20
    v_rows = m * p  # 30
    w_rows = p * n  # 24
    
    # We will collect terms as (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Or simpler: just build the matrices directly if we know the rank.
    # But rank is dynamic. Let's collect terms first.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))

    # Part 1: 4x4x4 multiplication (i,j,k in 0..3)
    # Using Strassen for 2x2 blocks
    
    # Strassen 2x2x2 decomposition
    # A indices: 0:11, 1:12, 2:21, 3:22 (local)
    # B indices: 0:11, 1:12, 2:21, 3:22 (local)
    # C indices: 0:11, 1:12, 2:21, 3:22 (local)
    
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    add_term([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1])
    # M2 = (A21+A22)B11 -> C21, C22
    add_term([2, 3], [1, 1], [0], [1], [2], [1]) # C21 gets +1
    # Wait, M2 contributes to C21 and C22?
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M2 has +1 in C21 and -1 in C22.
    # Let's re-verify Strassen formulas.
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 = (A11+A22)(B11+B22)
    #   U: A11(1), A22(1)
    #   V: B11(1), B22(1)
    #   W: C11(1), C22(1)
    
    # M2 = (A21+A22)B11
    #   U: A21(1), A22(1)
    #   V: B11(1)
    #   W: C21(1), C22(-1)
    
    # M3 = A11(B12-B22)
    #   U: A11(1)
    #   V: B12(1), B22(-1)
    #   W: C12(1), C22(1)
    
    # M4 = A22(B21-B11)
    #   U: A22(1)
    #   V: B21(1), B11(-1)
    #   W: C11(1), C21(1)
    
    # M5 = (A11+A12)B22
    #   U: A11(1), A12(1)
    #   V: B22(1)
    #   W: C11(-1), C12(1)
    
    # M6 = (A21-A11)(B11+B12)
    #   U: A21(1), A11(-1)
    #   V: B11(1), B12(1)
    #   W: C22(1)
    
    # M7 = (A12-A22)(B21+B22)
    #   U: A12(1), A22(-1)
    #   V: B21(1), B22(1)
    #   W: C11(1)
    
    # Let's clear the previous terms and add them correctly.
    terms = []
    
    # M1
    add_term([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1])
    # M2
    add_term([2, 3], [1, 1], [0], [1], [2, 3], [1, -1])
    # M3
    add_term([0], [1], [1, 3], [1, -1], [1, 3], [1, 1])
    # M4
    add_term([3], [1], [2, 0], [1, -1], [0, 2], [1, 1])
    # M5
    add_term([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1])
    # M6
    add_term([2, 0], [1, -1], [0, 1], [1, 1], [3], [1])
    # M7
    add_term([1, 3], [1, -1], [2, 3], [1, 1], [0], [1])
    
    # This is for a single 2x2x2 block.
    # Now we need to apply this to the 4x4x4 case.
    # The 4x4 matrix is composed of 2x2 blocks.
    # A blocks: A11 (i0,j0), A12 (i0,j1), A21 (i1,j0), A22 (i1,j1)
    # B blocks: B11 (j0,k0), B12 (j0,k1), B21 (j1,k0), B22 (j1,k1)
    # C blocks: C11 (i0,k0), C12 (i0,k1), C21 (i1,k0), C22 (i1,k1)
    
    # We need to map local indices (0..3) to global indices.
    # Local A index 0 (A11) corresponds to global A indices for i in 0..1, j in 0..1.
    # Wait, the Strassen algorithm operates on the *blocks*.
    # The "A11" in the Strassen formula refers to the top-left 2x2 block of the 4x4 matrix.
    # So, when we say "U index 0", it refers to the *entire block* A11?
    # No, the tensor decomposition is over *scalar* entries.
    # So, for the term corresponding to "A11" in the Strassen formula, we need to set U=1 for *all* scalar entries in the A11 block.
    
    # Let's define a helper to expand a block term to scalar terms.
    # A block is defined by i_block (0 or 1), j_block (0 or 1).
    # Scalar indices: i = 2*i_block + di, j = 2*j_block + dj.
    # U index = i * m + j.
    
    def expand_block(block_type, bi, bj, val):
        # block_type: 'A', 'B', 'C'
        # bi, bj: block indices (0 or 1)
        # Returns list of (global_index, val)
        indices = []
        if block_type == 'A':
            for di in range(2):
                for dj in range(2):
                    i = 2 * bi + di
                    j = 2 * bj + dj
                    indices.append((i * m + j, val))
        elif block_type == 'B':
            for di in range(2):
                for dj in range(2):
                    j = 2 * bi + di
                    k = 2 * bj + dj
                    indices.append((j * p + k, val))
        elif block_type == 'C':
            for di in range(2):
                for dj in range(2):
                # C index is k * n + i
                    k = 2 * bj + dj
                    i = 2 * bi + di
                    indices.append((k * n + i, val))
        return indices

    # Now, let's re-construct the terms for 4x4x4.
    terms_444 = []
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    # A11: block (0,0), A22: block (1,1)
    # B11: block (0,0), B22: block (1,1)
    # C11: block (0,0), C22: block (1,1)
    u_idx = expand_block('A', 0, 0, 1) + expand_block('A', 1, 1, 1)
    v_idx = expand_block('B', 0, 0, 1) + expand_block('B', 1, 1, 1)
    w_idx = expand_block('C', 0, 0, 1) + expand_block('C', 1, 1, 1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # M2: (A21+A22)B11 -> C21, C22
    # A21: block (1,0), A22: block (1,1)
    # B11: block (0,0)
    # C21: block (1,0), C22: block (1,1)
    u_idx = expand_block('A', 1, 0, 1) + expand_block('A', 1, 1, 1)
    v_idx = expand_block('B', 0, 0, 1)
    w_idx = expand_block('C', 1, 0, 1) + expand_block('C', 1, 1, -1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # M3: A11(B12-B22) -> C12, C22
    # A11: block (0,0)
    # B12: block (0,1), B22: block (1,1)
    # C12: block (0,1), C22: block (1,1)
    u_idx = expand_block('A', 0, 0, 1)
    v_idx = expand_block('B', 0, 1, 1) + expand_block('B', 1, 1, -1)
    w_idx = expand_block('C', 0, 1, 1) + expand_block('C', 1, 1, 1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # M4: A22(B21-B11) -> C11, C21
    # A22: block (1,1)
    # B21: block (1,0), B11: block (0,0)
    # C11: block (0,0), C21: block (1,0)
    u_idx = expand_block('A', 1, 1, 1)
    v_idx = expand_block('B', 1, 0, 1) + expand_block('B', 0, 0, -1)
    w_idx = expand_block('C', 0, 0, 1) + expand_block('C', 1, 0, 1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # M5: (A11+A12)B22 -> C11, C12
    # A11: block (0,0), A12: block (0,1)
    # B22: block (1,1)
    # C11: block (0,0), C12: block (0,1)
    u_idx = expand_block('A', 0, 0, 1) + expand_block('A', 0, 1, 1)
    v_idx = expand_block('B', 1, 1, 1)
    w_idx = expand_block('C', 0, 0, -1) + expand_block('C', 0, 1, 1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # M6: (A21-A11)(B11+B12) -> C22
    # A21: block (1,0), A11: block (0,0)
    # B11: block (0,0), B12: block (0,1)
    # C22: block (1,1)
    u_idx = expand_block('A', 1, 0, 1) + expand_block('A', 0, 0, -1)
    v_idx = expand_block('B', 0, 0, 1) + expand_block('B', 0, 1, 1)
    w_idx = expand_block('C', 1, 1, 1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # M7: (A12-A22)(B21+B22) -> C11
    # A12: block (0,1), A22: block (1,1)
    # B21: block (1,0), B22: block (1,1)
    # C11: block (0,0)
    u_idx = expand_block('A', 0, 1, 1) + expand_block('A', 1, 1, -1)
    v_idx = expand_block('B', 1, 0, 1) + expand_block('B', 1, 1, 1)
    w_idx = expand_block('C', 0, 0, 1)
    terms_444.append((u_idx, v_idx, w_idx))
    
    # Part 2: 4x4x2 multiplication (i,j in 0..3, k in 4..5)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_idx = [(i * m + j, 1)]
                v_idx = [(j * p + k, 1)]
                w_idx = [(k * n + i, 1)]
                terms_444.append((u_idx, v_idx, w_idx))
                
    # Part 3: 4x1x6 multiplication (i in 0..3, j in 4, k in 0..5)
    # Schoolbook
    for i in range(4):
        for k in range(6):
            u_idx = [(i * m + 4, 1)]
            v_idx = [(4 * p + k, 1)]
            w_idx = [(k * n + i, 1)]
            terms_444.append((u_idx, v_idx, w_idx))
            
    # Now build U, V, W matrices
    rank = len(terms_444)
    U = [[0] * rank for _ in range(u_rows)]
    V = [[0] * rank for _ in range(v_rows)]
    W = [[0] * rank for _ in range(w_rows)]
    
    for c, (u_idx, v_idx, w_idx) in enumerate(terms_444):
        for idx, val in u_idx:
            U[idx][c] = val
        for idx, val in v_idx:
            V[idx][c] = val
        for idx, val in w_idx:
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
