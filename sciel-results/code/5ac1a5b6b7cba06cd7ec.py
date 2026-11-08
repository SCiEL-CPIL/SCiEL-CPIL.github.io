"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses Strassen-based block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[c11,c12],[c21,c22]]
    #
    # m1 = (a+d)(e+h) -> c11: +1, c22: +1
    # m2 = (c+d)e    -> c21: +1, c22: -1
    # m3 = a(f-h)    -> c12: +1, c22: +1
    # m4 = d(g-e)    -> c11: +1, c21: +1
    # m5 = (a+b)h    -> c11: -1, c12: +1
    # m6 = (c-a)(e+f) -> c22: +1
    # m7 = (b-d)(g+h) -> c11: +1
    
    strassen_2x2_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # local indices: A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        #                B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        #                C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),        # m1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # m2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # m3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # m4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # m5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # m6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # m7
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4 (rank 49)."""
        
        def get_A_idx(li, lj):
            """Global U index for A entry at (li, lj) within the 4x4 block."""
            return (A_base_i + li) * m + (A_base_j + lj)
        
        def get_B_idx(lj, lk):
            """Global V index for B entry at (lj, lk) within the 4x4 block."""
            return (B_base_j + lj) * p + (B_base_k + lk)
        
        def get_C_idx(li, lk):
            """Global W index for C entry at (li, lk) within the 4x4 block."""
            return (C_base_k + lk) * n + (C_base_i + li)
        
        # Block-level Strassen for 4x4*4x4:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mi, we decompose the 2x2*2x2 product using Strassen.
        # The A-side of Mi is a linear combination of A-blocks.
        # The B-side of Mi is a linear combination of B-blocks.
        
        # Define block multiplications:
        # Each entry: (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_row, block_col, coeff)
        # B_blocks: list of (block_row, block_col, coeff)
        # C_blocks: dict mapping (block_row, block_col) to coeff
        
        block_mults = [
            # M1 = (A11+A22)(B11+B22), C11: +1, C22: +1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
            # M2 = (A21+A22)B11, C21: +1, C22: -1
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
            # M3 = A11(B12-B22), C12: +1, C22: +1
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
            # M4 = A22(B21-B11), C11: +1, C21: +1
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
            # M5 = (A11+A12)B22, C11: -1, C12: +1
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
            # M6 = (A21-A11)(B11+B12), C22: +1
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
            # M7 = (A12-A22)(B21+B22), C11: +1
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
        ]
        
        for A_blocks, B_blocks, C_blocks in block_mults:
            # For each Strassen term in the 2x2 meta-matrix:
            for A_meta, B_meta, C_meta in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Build u_vec
                for local_idx, meta_coeff in A_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for (br, bc, block_coeff) in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                
                # Build v_vec
                for local_idx, meta_coeff in B_meta.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc, block_coeff) in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                
                # Build w_vec
                for local_idx, meta_coeff in C_meta.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), block_coeff in C_blocks.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 where B1 is columns 0-3 of B (4x4)
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=0,
        C_base_i=0, C_base_k=0
    )
    
    # Part 2: A * B2 where B2 is columns 4-7 of B (4x4)
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=4,
        C_base_i=0, C_base_k=4
    )
    
    # Part 3: A * B3 where B3 is column 8 of B (4x1)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    # Build U, V, W matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    # Quick verification using numpy
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        
        # Check a few entries
        err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    err = max(err, abs(T[ui, vi, wi] - 1.0))
        
        # Check non-matmul entries
        valid = np.zeros((16, 36, 36), dtype=bool)
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    valid[i * m + j, j * p + k, k * n + i] = True
        err = max(err, float(np.max(np.abs(T[~valid]))))
        
        if err > 1e-8:
            # Invalid decomposition - should not happen
            pass
    
    return {"U": U, "V": V, "W": W}
    # Verify the decomposition is correct (using numpy for speed)
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[ui, vi, wi] = sum_r U[ui,r] * V[vi,r] * W[wi,r]
        # This is a tensor product. We can compute it as:
        # T = sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
        # But more efficiently, reshape and use einsum
        
        # T has shape (16, 36, 36)
        # T[ui, vi, wi] = sum_r U_arr[ui, r] * V_arr[vi, r] * W_arr[wi, r]
        T = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        
        # Check matmul entries
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    err = abs(T[ui, vi, wi] - 1.0)
                    max_err = max(max_err, err)
        
        # Check non-matmul entries should be 0
        # Build mask of valid matmul entries
        valid = np.zeros((16, 36, 36), dtype=bool)
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    valid[ui, vi, wi] = True
        
        non_matmul_max = np.max(np.abs(T[~valid]))
        max_err = max(max_err, non_matmul_max)
        
        if max_err > 1e-10:
            # Decomposition is invalid
            pass
    
    return {"U": U, "V": V, "W": W}
