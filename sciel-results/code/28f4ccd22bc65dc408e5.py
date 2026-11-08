"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses Strassen-based block decomposition:
  - Split B (4x9) into B1 (4x4) and B2 (4x5)
  - A*B1 uses Strassen for 4x4*4x4 (rank 49)
  - A*B2: split into B2a (4x4) Strassen (rank 49) + B2b (4x1) schoolbook (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 9  # A is n x m, B is m x p, C is n x p
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2*2x2 terms: (A_coeffs, B_coeffs, C_contribs)
    # Using (li,lj) for A, (lj,lk) for B, (li,lk) for C
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
                           A_blocks, B_blocks, C_blocks):
        """Add 7 Strassen terms for block-level 2x2*2x2 multiplication.
        
        A_blocks: list of (block_row, block_col, coeff) for A side
        B_blocks: list of (block_row, block_col, coeff) for B side  
        C_blocks: dict (block_row, block_col) -> coeff for C side
        """
        for A_coeffs, B_coeffs, C_contribs in strassen_terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for (li, lj), mc in A_coeffs.items():
                for (br, bc, bc_coeff) in A_blocks:
                    gi = A_i0 + br * 2 + li
                    gj = A_j0 + bc * 2 + lj
                    u_vec[gi * m + gj] += mc * bc_coeff
            
            for (lj, lk), mc in B_coeffs.items():
                for (br, bc, bc_coeff) in B_blocks:
                    gj = B_j0 + br * 2 + lj
                    gk = B_k0 + bc * 2 + lk
                    v_vec[gj * p + gk] += mc * bc_coeff
            
            for (li, lk), mc in C_contribs.items():
                for (br, bc), cc in C_blocks.items():
                    gi = C_i0 + br * 2 + li
                    gk = C_k0 + bc * 2 + lk
                    w_vec[gk * n + gi] += mc * cc
            
            add_term(u_vec, v_vec, w_vec)
    
    def strassen_4x4(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0):
        """Add 49 terms for 4x4 * 4x4 Strassen decomposition."""
        # M1 = (A11+A22)(B11+B22)
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)],
            {(0,0): 1, (1,1): 1})
        # M2 = (A21+A22)B11
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(1,0,1), (1,1,1)], [(0,0,1)],
            {(1,0): 1, (1,1): -1})
        # M3 = A11(B12-B22)
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(0,0,1)], [(0,1,1), (1,1,-1)],
            {(0,1): 1, (1,1): 1})
        # M4 = A22(B21-B11)
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(1,1,1)], [(1,0,1), (0,0,-1)],
            {(0,0): 1, (1,0): 1})
        # M5 = (A11+A12)B22
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(0,0,1), (0,1,1)], [(1,1,1)],
            {(0,0): -1, (0,1): 1})
        # M6 = (A21-A11)(B11+B12)
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],
            {(1,1): 1})
        # M7 = (A12-A22)(B21+B22)
        add_strassen_block(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0,
            [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],
            {(0,0): 1})
    
    def schoolbook_4x1(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0):
        """Add 16 terms for 4x4 * 4x1 schoolbook."""
        for i in range(4):
            for j in range(4):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                u_vec[i * m + j] = 1
                v_vec[j * p + B_k0] = 1
                w_vec[C_k0 * n + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B[:, 0:4] (4x4 * 4x4) -> C[:, 0:4]
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    if time.time() > deadline:
        # Fallback: return what we have plus schoolbook for rest
        for col in range(4, 9):
            schoolbook_4x1(0, 0, 0, col, 0, col)
        rank = len(U_cols)
        U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
        V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
        W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # Part 2: A * B[:, 4:8] (4x4 * 4x4) -> C[:, 4:8]
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    if time.time() > deadline:
        schoolbook_4x1(0, 0, 0, 8, 0, 8)
        rank = len(U_cols)
        U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
        V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
        W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # Part 3: A * B[:, 8] (4x4 * 4x1) -> C[:, 8]
    schoolbook_4x1(0, 0, 0, 8, 0, 8)
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    # Quick verification with random test
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_mat = np.array(U).T  # (rank, n*m)
        V_mat = np.array(V).T  # (rank, m*p)
        W_mat = np.array(W).T  # (rank, p*n)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        # C = sum_r U_r * V_r * W_r (outer products)
        C_reconstructed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_mat[r].reshape(n, m)
            v_r = V_mat[r].reshape(m, p)
            w_r = W_mat[r].reshape(p, n)
            # The tensor decomposition: C[i,k] = sum_r sum_j U[r,i,j] * V[r,j,k] * W[r,k,i]
            # Actually: C[i,k] = sum_r U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i] summed over j
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_mat[r, i * m + j] * V_mat[r, j * p + k] * W_mat[r, k * n + i]
                    C_reconstructed[i, k] += s
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            # Fallback to pure schoolbook if verification fails
            U_cols.clear()
            V_cols.clear()
            W_cols.clear()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        add_term(u_vec, v_vec, w_vec)
            rank = len(U_cols)
            U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
            V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
            W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
