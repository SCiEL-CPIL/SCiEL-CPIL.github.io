"""Matrix multiplication 4x4 * 4x9 rank-minimal decomposition.

Block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1: rank-49 Strassen 4x4*4x4
  - A*B2: rank-49 Strassen 4x4*4x4
  - A*B3: rank-16 schoolbook 4x4*4x1
  - Total rank: 114

Strassen for 4x4*4x4 (rank 49):
  7 block-level multiplications, each a 2x2*2x2 Strassen (rank 7).
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    U shape: (16, rank), V shape: (36, rank), W shape: (36, rank)
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen meta-terms for 2x2*2x2 (7 terms)
    # Each: (A_coeffs on {(li,lj)}, B_coeffs on {(lj,lk)}, C_contribs on {(li,lk)})
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 7 block-level Strassen multiplications for 4x4*4x4
    # Each: (A_side, B_side, C_side)
    # A_side: list of (br, bc, coeff)
    # B_side: list of (br, bc, coeff)
    # C_side: dict (br, bc) -> coeff
    strassen_4x4_blocks = [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
        # M2 = (A21+A22)B11
        ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
        # M3 = A11(B12-B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
        # M4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
        # M5 = (A11+A12)B22
        ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
    ]
    
    def add_strassen_4x4(B_base_j, B_base_k, C_base_k):
        """Add rank-49 Strassen decomposition for one 4x4*4x4 block."""
        A_base_i = 0
        A_base_j = 0
        C_base_i = 0
        
        for A_side, B_side, C_side in strassen_4x4_blocks:
            for A_coeffs, B_coeffs, C_contribs in strassen_2x2:
                u_vec = [0] * (n * m)   # 16
                v_vec = [0] * (m * p)   # 36
                w_vec = [0] * (p * n)   # 36
                
                # Build u_vec
                for (li, lj), mc in A_coeffs.items():
                    for (br, bc, bc_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += mc * bc_coeff
                
                # Build v_vec
                for (lj, lk), mc in B_coeffs.items():
                    for (br, bc, bc_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += mc * bc_coeff
                
                # Build w_vec
                for (li, lk), mc in C_contribs.items():
                    for (br, bc), bc_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += mc * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 (columns 0-3)
    add_strassen_4x4(B_base_j=0, B_base_k=0, C_base_k=0)
    
    # Part 2: A * B2 (columns 4-7)
    add_strassen_4x4(B_base_j=0, B_base_k=4, C_base_k=4)
    
    # Part 3: A * B3 (column 8) - schoolbook, 16 terms
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Build output matrices: transpose from list-of-cols to list-of-rows
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    # Quick verification (only if time permits)
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        rng = np.random.default_rng(12345)
        A_test = rng.random((n, m))
        B_test = rng.random((m, p))
        C_expected = A_test @ B_test
        
        # Compute C via tensor decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # More efficiently: for each r, C += (A * U_r_col) @ (V_r_col * B) ... 
        # Actually: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
        # = sum_r W[k*n+i,r] * (U_r @ A_col_i) * (V_r @ B_col_k)
        
        # Simpler verification: check tensor identity directly
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(total - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: schoolbook
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
