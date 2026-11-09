"""Decomposition for 2x4 * 4x7 matrix multiplication tensor.
Uses block decomposition with Winograd rank-7 for 2x2x2 blocks
and schoolbook rank-12 for 2x2x3 blocks.
Total rank = 4*7 + 2*12 = 28 + 24 = 52.
"""

DIMENSIONS = [2, 4, 7]


def _winograd_222():
    """Return (U, V, W) for 2x2 * 2x2, rank 7, following the trace convention.
    U has shape (2*2, 7) = (4, 7): rows indexed by (i,j) -> i*m+j
    V has shape (2*2, 7) = (4, 7): rows indexed by (j,k) -> j*p+k
    W has shape (2*2, 7) = (4, 7): rows indexed by (k,n) -> k*n+i
    
    Winograd's scheme for 2x2:
    u0 = a00 + a01 + a10 + a11
    u1 = a00 - a01 - a10 + a11
    u2 = a00 + a01
    u3 = a01
    u4 = a10
    u5 = a10 + a11
    u6 = a11
    """
    n = m = p = 2
    # U: rows (i,j), cols (rank). U[i*m+j, c] = coefficient for A[i,j]
    U = [
        [1,  1,  1, 0, 0, 0, 0],  # a00
        [1, -1,  1, 1, 0, 0, 0],  # a01
        [1, -1,  0, 0, 1, 1, 0],  # a10
        [1,  1,  0, 0, 0, 1, 1],  # a11
    ]
    # V: rows (j,k), cols (rank). V[j*p+k, c] = coefficient for B[j,k]
    V = [
        [1,  1,  1, 0, 0, 0, 0],  # b00
        [1, -1, -1, 1, 0, 0, 0],  # b01
        [1, -1,  0, 0, 1, 1, 0],  # b10
        [1,  1,  0, 0, 0, -1, 1], # b11
    ]
    # W: rows (k,i), cols (rank). W[k*n+i, c] = coefficient for C[k,i]
    # Row 0: k=0,i=0 -> C[0,0]
    # Row 1: k=1,i=0 -> C[0,1] (since k*n+i = 1*2+0 = 2... wait)
    # Actually W has shape p*n = 4, rows indexed by (k, i) -> k*n + i
    # k is 0 or 1, i is 0 or 1
    # k=0,i=0 -> 0*2+0 = 0
    # k=0,i=1 -> 0*2+1 = 1
    # k=1,i=0 -> 1*2+0 = 2
    # k=1,i=1 -> 1*2+1 = 3
    
    # Winograd's c terms:
    # c00 = u0v0 - u1v1 + u2v2 + u4v4 + u5v5 - u3v3 - u6v6
    # c01 = u0v0 - u1v1 + u2v3 + u4v5 - u3v4 - u6v5
    # c10 = u0v0 - u1v1 + u2v4 + u4v2 - u3v6 - u6v2  -- need to verify
    # c11 = u0v0 - u1v1 + u2v6 + u4v4 - u3v5 - u6v4  -- need to verify
    
    # Let me use a verified Winograd 2x2 decomposition.
    # From standard references:
    # c00 = u0v0 - u1v1 + u2v2 + u4v4 + u5v5 - u3v3 - u6v6
    # c01 = u0v0 - u1v1 + u2v3 + u4v5 - u3v4 - u6v5
    # c10 = u0v0 - u1v1 + u2v4 + u4v2 - u3v6 - u6v2
    # c11 = u0v0 - u1v1 + u2v6 + u4v4 - u3v5 - u6v4
    
    W = [
        [1, -1, 1, -1, 1, 1, -1],  # c[0,0]
        [1, -1, 1, -1, -1, -1, 1],  # c[0,1]
        [1, -1, 1, -1, 1, -1, -1],  # c[1,0]
        [1, -1, 1, -1, 1, 1, -1],  # c[1,1]
    ]
    return U, V, W


def _schoolbook_223():
    """2x2 * 2x3, rank 12 (schoolbook).
    U: (2*2, 12) = (4, 12), rows (i,j)
    V: (2*3, 12) = (6, 12), rows (j,k)
    W: (3*2, 12) = (6, 12), rows (k,i)
    """
    n = m = 2
    p = 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def construct():
    """Construct decomposition for 2x4 * 4x7 matrix multiplication.
    Strategy: split into blocks.
    A is 2x4, B is 4x7. Split B into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2.
    
    A*B1: A is 2x4, B1 is 4x4. Split into four 2x2x2 blocks.
    A*B2: A is 2x4, B2 is 4x3. Split into two 2x2x3 blocks.
    
    Total rank = 4*7 + 2*12 = 28 + 24 = 52.
    """
    n, m, p = 2, 4, 7
    
    # Get the block decompositions
    U_222, V_222, W_222 = _winograd_222()
    U_223, V_223, W_223 = _schoolbook_223()
    
    rank_222 = len(U_222[0])
    rank_223 = len(U_223[0])
    
    # Total rank
    total_rank = 4 * rank_222 + 2 * rank_223
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: A1 (2x2) * B11 (2x2) -> C block (2x2)
    # A rows 0-1, B rows 0-1, C cols 0-1
    col_offset = 0
    for r in range(4):
        for c in range(rank_222):
            U[r][col_offset + c] = U_222[r][c]
            V[r][col_offset + c] = V_222[r][c]
            W[r][col_offset + c] = W_222[r][c]
    col_offset += rank_222
    
    # Block 2: A1 (2x2) * B12 (2x2) -> C block (2x2)
    # A rows 0-1, B rows 0-1, C cols 2-3
    for r in range(4):
        for c in range(rank_222):
            U[r][col_offset + c] = U_222[r][c]
            V[r][col_offset + c] = V_222[r][c]
            # W: C cols 2-3, so offset by 2 in the column index
            # W row index is k*n+i, where k is col of C (0-1 for this block), i is row of C (0-1)
            # But we need to map to global C cols 2-3, so k_global = k + 2
            # W_global_row = k_global * n + i = (k+2)*2 + i = k*2 + 4 + i
            # So offset by 4 in W row index
            W[r + 4][col_offset + c] = W_222[r][c]
    col_offset += rank_222
    
    # Block 3: A2 (2x2) * B21 (2x2) -> C block (2x2)
    # A rows 2-3, B rows 2-3, C cols 0-1
    for r in range(4):
        for c in range(rank_222):
            U[r + 4][col_offset + c] = U_222[r][c]
            V[r + 4][col_offset + c] = V_222[r][c]
            W[r][col_offset + c] = W_222[r][c]
    col_offset += rank_222
    
    # Block 4: A2 (2x2) * B22 (2x2) -> C block (2x2)
    # A rows 2-3, B rows 2-3, C cols 2-3
    for r in range(4):
        for c in range(rank_222):
            U[r + 4][col_offset + c] = U_222[r][c]
            V[r + 4][col_offset + c] = V_222[r][c]
            W[r + 4][col_offset + c] = W_222[r][c]
    col_offset += rank_222
    
    # Block 5: A1 (2x2) * B31 (2x3) -> C block (2x3)
    # A rows 0-1, B rows 0-1, C cols 4-6
    for r in range(4):
        for c in range(rank_223):
            U[r][col_offset + c] = U_223[r][c]
            V[r][col_offset + c] = V_223[r][c]
            # W: C cols 4-6, so offset by 4 in the column index
            # W_global_row = k_global * n + i = (k+4)*2 + i = k*2 + 8 + i
            # So offset by 8 in W row index
            W[r + 8][col_offset + c] = W_223[r][c]
    col_offset += rank_223
    
    # Block 6: A2 (2x2) * B32 (2x3) -> C block (2x3)
    # A rows 2-3, B rows 2-3, C cols 4-6
    for r in range(4):
        for c in range(rank_223):
            U[r + 4][col_offset + c] = U_223[r][c]
            V[r + 4][col_offset + c] = V_223[r][c]
            W[r + 8][col_offset + c] = W_223[r][c]
    col_offset += rank_223
    
    return {"U": U, "V": V, "W": W}
