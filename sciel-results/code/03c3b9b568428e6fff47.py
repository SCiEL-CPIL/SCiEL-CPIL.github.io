"""Matrix multiplication 4,4,8 - Strassen-based rank-98 decomposition."""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def construct():
    """
    Returns a rank-98 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    
    Uses Strassen's 2x2x2 algorithm (rank 7) recursively:
    - Split B (4x8) into two 4x4 halves B1, B2
    - C1 = A*B1 (4x4), C2 = A*B2 (4x4)
    - Each 4x4x4 uses recursive Strassen: 7 * 7 = 49 rank-1 terms
    - Total: 2 * 49 = 98 rank-1 terms
    """
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2: 7 terms
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 4 values for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_coeffs: 4 values for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_coeffs: 4 values for C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_2x2 = [
        # M1 = (a00+a11)(b00+b11), contributes to c00 (+1), c11 (+1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a10+a11)(b00), contributes to c10 (+1), c11 (-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = (a00)(b01-b11), contributes to c01 (+1), c11 (+1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = (a11)(b10-b00), contributes to c00 (+1), c10 (+1)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5 = (a00+a01)(b11), contributes to c00 (-1), c01 (+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a10-a00)(b00+b01), contributes to c11 (+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a01-a11)(b10+b11), contributes to c00 (+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Outer Strassen: 7 terms for 4x4x4 with 2x2 block elements
    # Each term: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # Block order: 00, 01, 10, 11
    outer_strassen = [
        # M1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A10+A11)(B00) -> C10:+1, C11:-1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # M3 = (A00)(B01-B11) -> C01:+1, C11:+1
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        # M4 = (A11)(B10-B00) -> C00:+1, C10:+1
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        # M5 = (A00+A01)(B11) -> C00:-1, C01:+1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6 = (A10-A00)(B00+B01) -> C11:+1
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A01-A11)(B10+B11) -> C00:+1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Helper: get element indices for a 2x2 block within a 4x4 matrix
    def block_elements(bi, bj):
        """Return list of (i,j) positions for block (bi,bj) in 4x4 matrix."""
        return [(bi*2 + di, bj*2 + dj) for di in range(2) for dj in range(2)]
    
    # Build the 49 rank-1 terms for 4x4x4
    def build_4x4x4_terms():
        terms = []
        for outer_idx, (ao, bo, co) in enumerate(outer_strassen):
            for inner_idx, (ai, bi, ci) in enumerate(strassen_2x2):
                u = [0] * 16  # 4x4 A
                v = [0] * 16  # 4x4 B
                w = [0] * 16  # 4x4 C
                
                # For each block of A
                for block_idx in range(4):
                    if ao[block_idx] == 0:
                        continue
                    bi_, bj_ = block_idx // 2, block_idx % 2
                    elems = block_elements(bi_, bj_)
                    for local_idx, (ei, ej) in enumerate(elems):
                        u[ei * 4 + ej] = ao[block_idx] * ai[local_idx]
                
                # For each block of B
                for block_idx in range(4):
                    if bo[block_idx] == 0:
                        continue
                    bi_, bj_ = block_idx // 2, block_idx % 2
                    elems = block_elements(bi_, bj_)
                    for local_idx, (ei, ej) in enumerate(elems):
                        v[ei * 4 + ej] = bo[block_idx] * bi[local_idx]
                
                # For each block of C
                for block_idx in range(4):
                    if co[block_idx] == 0:
                        continue
                    bi_, bj_ = block_idx // 2, block_idx % 2
                    elems = block_elements(bi_, bj_)
                    for local_idx, (ei, ej) in enumerate(elems):
                        w[ei * 4 + ej] = co[block_idx] * ci[local_idx]
                
                terms.append((u, v, w))
        
        return terms
    
    # Build full 4x4x8 decomposition
    # A is 4x4 (16 elements), B is 4x8 (32 elements), C is 4x8 (32 elements)
    # U shape: (16, rank), V shape: (32, rank), W shape: (32, rank)
    # W[k*n+i, r] where k is col of C (0..7), i is row of C (0..3)
    
    terms_4x4x4 = build_4x4x4_terms()
    rank = 98
    
    U = np.zeros((16, rank), dtype=np.int8)
    V = np.zeros((32, rank), dtype=np.int8)
    W = np.zeros((32, rank), dtype=np.int8)
    
    col = 0
    for half in range(2):
        for (u, v, w) in terms_4x4x4:
            # U: A is the same for both halves
            for i in range(16):
                U[i, col] = u[i]
            
            # V: B_half elements mapped to B (4x8)
            # B_half has elements (i,j) for i in 0..3, j in 0..3
            # In B (4x8), element (i, j) is at position i*8 + (j + half*4)
            for i in range(4):
                for j in range(4):
                    b_idx = i * 8 + (j + half * 4)
                    v_local_idx = i * 4 + j
                    V[b_idx, col] = v[v_local_idx]
            
            # W: C_half elements mapped to C (4x8)
            # W[k*n+i, r] where k is col of C (0..7), i is row of C (0..3)
            for i in range(4):  # row of C
                for j in range(4):  # col of C_half
                    k_global = j + half * 4  # global col of C
                    w_row_idx = k_global * 4 + i
                    w_local_idx = i * 4 + j
                    W[w_row_idx, col] = w[w_local_idx]
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

