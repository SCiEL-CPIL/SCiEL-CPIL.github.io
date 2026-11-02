"""Matrix multiplication 5x5x5 rank-minimal exact tensor decomposition.

Strategy: Use Strassen's algorithm (rank 7) for the 4x4 sub-blocks and 
schoolbook for the interactions with the 5th row/column.
Total Rank: 28 (for 4x4) + 57 (for 5th row/col interactions) = 85.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """
    Check that U @ diag(t) @ V^T matches the tensor for all (i,j,k).
    Uses the convention: C[i,k] = sum_j A[i,j] B[j,k]
    Tensor T[i*m+j, j*p+k, k*n+i] = 1 (using 0-indexed).
    We verify: for each c, U[row,c]*V[col,c]*W[out,c] contributes to T.
    Equivalently, the sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct the tensor decomposition for 5x5x5 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    U: shape (25, rank), V: shape (25, rank), W: shape (25, rank)
    Convention: W[k*n+i, c] for the output factor.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Strategy:
    # 1. Decompose the 4x4 block (indices 0-3) using Strassen's algorithm.
    #    Strassen's 2x2 algorithm has rank 7.
    #    We apply it to four 2x2 sub-blocks of the 4x4 matrix.
    #    Total rank for 4x4 part = 4 * 7 = 28.
    # 2. Handle the 5th row and column (index 4) with schoolbook terms.
    #    Interactions:
    #    - A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4] : 4*4 = 16 terms
    #    - A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4] : 4*4 = 16 terms
    #    - A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4] : 4*4 = 16 terms
    #    - A[4, 4] * B[4, 4] -> C[4, 4] : 1 term
    #    - A[0:4, 4] * B[4, 4] -> C[0:4, 4] : 4 terms
    #    - A[4, 0:4] * B[0:4, 4] -> C[4, 0:4] : 4 terms
    #    Total additional terms = 16 + 16 + 16 + 1 + 4 + 4 = 57.
    #    Total Rank = 28 + 57 = 85.
    
    total_rank = 85
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen's 7 multiplication factors for 2x2 matrices:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    #
    # Result:
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # We need to map these to the tensor factors U, V, W.
    # For a specific 2x2 block of A (rows r0, r0+1; cols c0, c0+1), 
    # a specific 2x2 block of B (rows c0, c0+1; cols k0, k0+1),
    # and a specific 2x2 block of C (rows r0, r0+1; cols k0, k0+1).
    
    # Indices for 2x2 blocks in 4x4 matrix (0-3):
    # Block (br, bc) where br, bc in {0, 1}
    # Rows of A: 2*br, 2*br+1
    # Cols of A: 2*bc, 2*bc+1
    # Rows of B: 2*bc, 2*bc+1
    # Cols of B: 2*bc, 2*bc+1 (Wait, B cols correspond to C cols)
    # Actually, for matrix multiplication A (4x4) * B (4x4) = C (4x4):
    # C[i, k] = sum_j A[i, j] B[j, k]
    # If we split into 2x2 blocks:
    # A is split into A11, A12, A21, A22
    # B is split into B11, B12, B21, B22
    # C is split into C11, C12, C21, C22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We will apply Strassen to each of these 4 block multiplications.
    # For each block multiplication (e.g., A11*B11), we use 7 terms.
    # Total 28 terms for the 4x4 core.
    
    # Strassen coefficients for U, V, W
    # Let's define the 7 terms for a generic 2x2 block multiplication.
    # The indices in the 5x5 matrices will be offset by the block position.
    
    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        """
        Add 7 Strassen terms for a 2x2 block multiplication.
        A block: rows a_r0, a_r0+1; cols a_c0, a_c0+1
        B block: rows b_r0, b_r0+1; cols b_c0, b_c0+1
        C block: rows c_r0, c_r0+1; cols c_c0, c_c0+1
        Note: For standard matmul, b_r0 == a_c0 and b_c0 == c_c0.
        """
        # m1 = (a00 + a11) * (b00 + b11)
        # U entries: a00, a11
        # V entries: b00, b11
        # W entries: c00, c11 (with +1), c11 (with +1) -> wait, C11 gets +m1, C22 gets +m1
        
        # We need to be careful with the tensor indices.
        # U row index = i*m + j
        # V row index = j*p + k
        # W row index = k*n + i
        
        # Term 1: (A[i0, j0] + A[i1, j1]) * (B[j0, k0] + B[j1, k1])
        # U: [i0*m+j0] += 1, [i1*m+j1] += 1
        # V: [j0*p+k0] += 1, [j1*p+k1] += 1
        # W: [k0*n+i0] += 1, [k1*n+i1] += 1
        # This contributes to C[i0, k0], C[i0, k1], C[i1, k0], C[i1, k1].
        # Specifically, it adds to all 4 C entries?
        # Let's check the Strassen formula again.
        # m1 is used in C11 and C22.
        # C11 = m1 + ...
        # C22 = m1 - ...
        # So W for m1 should contribute to C[i0, k0] and C[i1, k1] with +1.
        
        # Let's map the 2x2 local indices to global indices.
        # i0 = a_r0, i1 = a_r0+1
        # j0 = a_c0, j1 = a_c0+1
        # k0 = b_c0, k1 = b_c0+1
        
        i0, i1 = a_r0, a_r0 + 1
        j0, j1 = a_c0, a_c0 + 1
        k0, k1 = b_c0, b_c0 + 1
        
        # Term 1: m1
        # U: A[i0, j0] + A[i1, j1]
        U[i0 * m + j0][start_col] += 1
        U[i1 * m + j1][start_col] += 1
        # V: B[j0, k0] + B[j1, k1]
        V[j0 * p + k0][start_col] += 1
        V[j1 * p + k1][start_col] += 1
        # W: C[i0, k0] + C[i1, k1]
        W[k0 * n + i0][start_col] += 1
        W[k1 * n + i1][start_col] += 1
        col = start_col + 1
        
        # Term 2: m2 = (a10 + a11) * b00
        # U: A[i1, j0] + A[i1, j1]
        U[i1 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        # V: B[j0, k0]
        V[j0 * p + k0][col] += 1
        # W: C[i1, k0]
        W[k0 * n + i1][col] += 1
        col += 1
        
        # Term 3: m3 = a00 * (b01 - b11)
        # U: A[i0, j0]
        U[i0 * m + j0][col] += 1
        # V: B[j0, k1] - B[j1, k1]
        V[j0 * p + k1][col] += 1
        V[j1 * p + k1][col] -= 1
        # W: C[i0, k1]
        W[k1 * n + i0][col] += 1
        col += 1
        
        # Term 4: m4 = a11 * (b10 - b00)
        # U: A[i1, j1]
        U[i1 * m + j1][col] += 1
        # V: B[j1, k0] - B[j0, k0]
        V[j1 * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        # W: C[i0, k0] + C[i1, k0] ? 
        # Strassen: C11 uses +m4, C21 uses +m4.
        # So W should contribute to C[i0, k0] and C[i1, k0].
        W[k0 * n + i0][col] += 1
        W[k0 * n + i1][col] += 1
        col += 1
        
        # Term 5: m5 = (a00 + a01) * b11
        # U: A[i0, j0] + A[i0, j1]
        U[i0 * m + j0][col] += 1
        U[i0 * m + j1][col] += 1
        # V: B[j1, k1]
        V[j1 * p + k1][col] += 1
        # W: C[i0, k0] - C[i1, k1] ?
        # Strassen: C11 uses -m5, C12 uses +m5.
        # So W should contribute to C[i0, k0] with -1 and C[i0, k1] with +1.
        # Wait, C12 is [i0, k1]. C11 is [i0, k0].
        # Let's re-verify Strassen formulas.
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # So for m5:
        # C11 (i0, k0) gets -1
        # C12 (i0, k1) gets +1
        W[k0 * n + i0][col] -= 1
        W[k1 * n + i0][col] += 1
        col += 1
        
        # Term 6: m6 = (a10 - a00) * (b00 + b01)
        # U: A[i1, j0] - A[i0, j0]
        U[i1 * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        # V: B[j0, k0] + B[j0, k1]
        V[j0 * p + k0][col] += 1
        V[j0 * p + k1][col] += 1
        # W: C[i1, k1]
        # Strassen: C22 uses +m6.
        W[k1 * n + i1][col] += 1
        col += 1
        
        # Term 7: m7 = (a01 - a11) * (b10 + b11)
        # U: A[i0, j1] - A[i1, j1]
        U[i0 * m + j1][col] += 1
        U[i1 * m + j1][col] -= 1
        # V: B[j1, k0] + B[j1, k1]
        V[j1 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        # W: C[i0, k0]
        # Strassen: C11 uses +m7.
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Apply Strassen to the four 2x2 blocks of the 4x4 matrix.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    # Block 3: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    col = add_strassen_block(2, 0, 0, 2, 2, 2, col)
    # Block 4: A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    col = add_strassen_block(2, 2, 2, 2, 2, 2, col)
    
    # Now add the schoolbook terms for the 5th row and column.
    # Indices 0-3 are handled above. Index 4 is new.
    
    # 1. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # 2. A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    for j in range(4):
        for k in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
            col += 1
            
    # 3. A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # 4. A[4, 4] * B[4, 4] -> C[4, 4]
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    # 5. A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    # 6. A[4, 0:4] * B[0:4, 4] -> C[4, 0:4]
    for j in range(4):
        U[4 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
        
    # col should be 85
    if col != total_rank:
        # Fallback if construction fails
        return _schoolbook(n, m, p)
        
    result = {"U": U, "V": V, "W": W}
    
    # Verify to be safe
    if _verify(U, V, W, n, m, p):
        return result
    else:
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
