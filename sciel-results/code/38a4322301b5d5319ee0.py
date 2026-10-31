import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor.
    Uses recursive Strassen algorithm: 7 multiplications for 2x2 blocks,
    applied to the 4 blocks of the 4x4 matrices, yielding 7*7=49 terms.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Strassen 2x2 decomposition coefficients
    # Each term is (A_indices, B_indices, C_indices)
    # A_indices: list of (row, col, coeff)
    # We use 0-based indexing for 2x2: (0,0), (0,1), (1,0), (1,1)
    
    # Standard Strassen for C = AB:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00)     -> C10 += M2, C11 -= M2
    # M3 = (A00)(B01-B11)     -> C01 += M3, C11 += M3
    # M4 = (A11)(B10-B00)     -> C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11)     -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    strassen_2x2 = [
        # A part, B part, C part
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],         [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],         [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),
        ([(1,1,1)],         [(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)],         [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)], [(0,0,1)])
    ]

    # The 4x4 matrix multiplication is a 2x2 matrix multiplication over the ring of 2x2 matrices.
    # We apply the same Strassen structure at the "block" level.
    # Block indices: 00, 01, 10, 11.
    # Block (br, bc) of the 4x4 matrix corresponds to a 2x2 submatrix.
    # Global row/col for A: r = br*2 + lr, c = bc*2 + lc
    # U index: i*4 + j. Here i=r, j=c.
    # V index: j*4 + k. Here j=r, k=c.
    # W index: k*4 + i. Here k=c, i=r. (Note: W[k*n+i] where n=4, so W index = col_C * 4 + row_C)
    
    terms = []  # List of (U_coeffs, V_coeffs, W_coeffs) where coeffs are dicts {global_index: value}
    
    # Outer Strassen operates on 2x2 blocks of 2x2 matrices.
    # The structure is identical to the inner Strassen.
    # We just need to map local block indices to global indices.
    
    # For each of the 7 outer terms, we have linear combinations of A-blocks and B-blocks.
    # For each of the 7 inner terms, we have linear combinations of elements within those blocks.
    
    # Let's iterate over outer terms and inner terms.
    
    for outer_term in strassen_2x2:
        a_blocks, b_blocks, c_blocks = outer_term
        # a_blocks: list of (block_row, block_col, coeff) for A
        # b_blocks: list of (block_row, block_col, coeff) for B
        # c_blocks: list of (block_row, block_col, coeff) for C
        
        for inner_term in strassen_2x2:
            a_elems, b_elems, c_elems = inner_term
            # a_elems: list of (local_row, local_col, coeff) within the 2x2 block
            # b_elems: list of (local_row, local_col, coeff) within the 2x2 block
            # c_elems: list of (local_row, local_col, coeff) within the 2x2 block
            
            U_coeffs = {}
            V_coeffs = {}
            W_coeffs = {}
            
            # Build U part:
            # U is indexed by (i, j) where i is row of A, j is col of A.
            # i = br*2 + lr, j = bc*2 + lc
            # Index in U matrix: i*4 + j
            for (lr, lc, ca) in a_elems:
                for (br, bc, cb) in a_blocks:
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    idx = i * 4 + j
                    U_coeffs[idx] = U_coeffs.get(idx, 0) + ca * cb
            
            # Build V part:
            # V is indexed by (j, k) where j is row of B, k is col of B.
            # j = br*2 + lr, k = bc*2 + lc
            # Index in V matrix: j*4 + k
            for (lr, lc, cb) in b_elems:
                for (br, bc, cc) in b_blocks:
                    j = br * 2 + lr
                    k = bc * 2 + lc
                    idx = j * 4 + k
                    V_coeffs[idx] = V_coeffs.get(idx, 0) + cb * cc
            
            # Build W part:
            # W is indexed by (k, i) where k is col of C, i is row of C.
            # k = bc*2 + lc, i = br*2 + lr
            # Index in W matrix: k*4 + i
            for (lr, lc, cc) in c_elems:
                for (br, bc, cd) in c_blocks:
                    k = bc * 2 + lc
                    i = br * 2 + lr
                    idx = k * 4 + i
                    W_coeffs[idx] = W_coeffs.get(idx, 0) + cc * cd
            
            terms.append((U_coeffs, V_coeffs, W_coeffs))
            
            # Check time
            if time.time() > deadline:
                # Fallback to schoolbook if we somehow timeout (unlikely)
                return _schoolbook_4x4()
    
    rank = len(terms)
    if rank != 49:
        return _schoolbook_4x4()
        
    # Initialize matrices
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    for col, (u_d, v_d, w_d) in enumerate(terms):
        for idx, val in u_d.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_d.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_d.items():
            if val != 0:
                W[idx][col] = val
    
    # Quick verification with NumPy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    # Let's check a few random points or all points (64 checks is fast)
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                # Vectorized sum over r
                # U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :]
                prod = U_np[i*4+j] * V_np[j*4+k] * W_np[k*4+i]
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        return _schoolbook_4x4()
        
    return {"U": U, "V": V, "W": W}

def _schoolbook_4x4():
    """Fallback to rank 64 schoolbook multiplication."""
    rank = 64
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1.0
                V[j*4+k][col] = 1.0
                W[k*4+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
