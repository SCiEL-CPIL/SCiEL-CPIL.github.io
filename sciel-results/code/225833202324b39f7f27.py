"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    n=2, m=2, p=2.
    U shape: (n*m, rank) = (4, 7)
    V shape: (m*p, rank) = (4, 7)
    W shape: (p*n, rank) = (4, 7)
    
    Indices:
    U[i*m + j, k]
    V[j*p + k, l]
    W[k'*n + i', m']  -> W[(j*p + k)*n + i] in schoolbook, but here we follow trace convention W[k*n+i, r]
    Wait, the problem states: "The output factor follows the official trace convention W[k*n+i,r]."
    This means for the full problem, W has shape (p*n, rank). The row index is k*n + i.
    In Strassen, the output C[i,k] = sum_j A[i,j]B[j,k].
    So the row index in W is k*n + i.
    
    Let's define the 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U (A coefficients):
    Row 0: A00. Row 1: A01. Row 2: A10. Row 3: A11.
    Col 0 (M1): A00(1), A11(1) -> [1, 0, 0, 1]
    Col 1 (M2): A10(1), A11(1) -> [0, 0, 1, 1]
    Col 2 (M3): A00(1) -> [1, 0, 0, 0]
    Col 3 (M4): A11(1) -> [0, 0, 0, 1]
    Col 4 (M5): A00(1), A01(1) -> [1, 1, 0, 0]
    Col 5 (M6): A10(1), A00(-1) -> [-1, 0, 1, 0]
    Col 6 (M7): A01(1), A11(-1) -> [0, 1, 0, -1]
    
    V (B coefficients):
    Row 0: B00. Row 1: B01. Row 2: B10. Row 3: B11.
    Col 0 (M1): B00(1), B11(1) -> [1, 0, 0, 1]
    Col 1 (M2): B00(1) -> [1, 0, 0, 0]
    Col 2 (M3): B10(1), B11(-1) -> [0, 0, 1, -1]
    Col 3 (M4): B01(1), B00(-1) -> [-1, 1, 0, 0]
    Col 4 (M5): B11(1) -> [0, 0, 0, 1]
    Col 5 (M6): B00(1), B01(1) -> [1, 1, 0, 0]
    Col 6 (M7): B10(1), B11(1) -> [0, 0, 1, 1]
    
    W (C coefficients):
    Row 0: C00. Row 1: C01. Row 2: C10. Row 3: C11.
    Col 0 (M1): C00(1), C11(1) -> [1, 0, 0, 1]
    Col 1 (M2): C10(1), C11(-1) -> [0, 0, 1, -1]
    Col 2 (M3): C01(1), C11(1) -> [0, 1, 0, 1]
    Col 3 (M4): C00(1), C10(1) -> [1, 0, 1, 0]
    Col 4 (M5): C00(-1), C01(1) -> [-1, 1, 0, 0]
    Col 5 (M6): C11(1) -> [0, 0, 0, 1]
    Col 6 (M7): C00(-1) -> [-1, 0, 0, 0]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Returns U, V, W for nxmxp matrix multiplication with schoolbook rank n*m*p.
    """
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
    return U, V, W

def _embed(U_sub, V_sub, W_sub, i0, j0, k0, n, m, p, rank):
    """
    Embeds sub-decompositions into global U, V, W.
    i0, j0, k0 are the starting indices for the sub-block in the global matrix.
    n, m, p are the dimensions of the sub-block.
    Returns lists of (row_idx, col_idx, val) for each matrix.
    """
    U_entries = []
    V_entries = []
    W_entries = []
    
    # Determine sub-block sizes
    # The sub-block is from i0 to i0+n, j0 to j0+m, k0 to k0+p
    
    for c in range(rank):
        # U
        for r in range(n * m):
            val = U_sub[r][c]
            if val != 0:
                # Global row index: (i0 + i_local) * (global_m) + (j0 + j_local)
                # But we don't know global_m here. We need to pass global dimensions or return local indices.
                # Let's return local indices and let the caller map them.
                pass
        # V
        for r in range(m * p):
            val = V_sub[r][c]
            if val != 0:
                pass
        # W
        for r in range(p * n):
            val = W_sub[r][c]
            if val != 0:
                pass
    return [], [], []

def _construct_block(i0, j0, k0, n, m, p, global_n, global_m, global_p):
    """
    Constructs U, V, W entries for a specific block.
    Returns (U_entries, V_entries, W_entries)
    U_entries: list of (global_row, col, val)
    V_entries: list of (global_row, col, val)
    W_entries: list of (global_row, col, val)
    """
    # Decide which decomposition to use
    if n == 2 and m == 2 and p == 2:
        U_sub, V_sub, W_sub = _strassen_222()
    else:
        U_sub, V_sub, W_sub = _schoolbook_nm_p(n, m, p)
    
    rank = len(U_sub[0])
    U_entries = []
    V_entries = []
    W_entries = []
    
    for c in range(rank):
        # U
        for r in range(n * m):
            val = U_sub[r][c]
            if val != 0:
                i_local = r // m
                j_local = r % m
                i_global = i0 + i_local
                j_global = j0 + j_local
                u_row = i_global * global_m + j_global
                U_entries.append((u_row, c, val))
        
        # V
        for r in range(m * p):
            val = V_sub[r][c]
            if val != 0:
                j_local = r // p
                k_local = r % p
                j_global = j0 + j_local
                k_global = k0 + k_local
                v_row = j_global * global_p + k_global
                V_entries.append((v_row, c, val))
        
        # W
        for r in range(p * n):
            val = W_sub[r][c]
            if val != 0:
                k_local = r // n
                i_local = r % n
                k_global = k0 + k_local
                i_global = i0 + i_local
                w_row = k_global * global_n + i_global
                W_entries.append((w_row, c, val))
    
    return U_entries, V_entries, W_entries

def construct():
    n, m, p = DIMENSIONS
    # 4x5x5
    # Partition into:
    # Block 1: i=0,1; j=0,1; k=0,1 (2x2x2) -> Strassen (7)
    # Block 2: i=0,1; j=2,3; k=0,1 (2x2x2) -> Strassen (7)
    # Block 3: i=0,1; j=4;   k=0,1 (2x1x2) -> Schoolbook (4)
    # Block 4: i=2,3; j=0,1; k=0,1 (2x2x2) -> Strassen (7)
    # Block 5: i=2,3; j=2,3; k=0,1 (2x2x2) -> Strassen (7)
    # Block 6: i=2,3; j=4;   k=0,1 (2x1x2) -> Schoolbook (4)
    # Remaining k=2,3,4
    # Block 7: i=0,1; j=0,1; k=2,3,4 (2x2x3)
    # Block 8: i=0,1; j=2,3; k=2,3,4 (2x2x3)
    # Block 9: i=0,1; j=4;   k=2,3,4 (2x1x3)
    # Block 10: i=2,3; j=0,1; k=2,3,4 (2x2x3)
    # Block 11: i=2,3; j=2,3; k=2,3,4 (2x2x3)
    # Block 12: i=2,3; j=4;   k=2,3,4 (2x1x3)
    
    # We need to decompose 2x2x3 and 2x1x3.
    # 2x2x3: Split k into 2+1.
    #   2x2x2 (Strassen, 7) + 2x2x1 (Schoolbook, 4) = 11
    # 2x1x3: Split k into 2+1.
    #   2x1x2 (Schoolbook, 4) + 2x1x1 (Schoolbook, 2) = 6
    
    # Total Rank:
    # 4 * 7 (Strassen 2x2x2) = 28
    # 2 * 4 (Schoolbook 2x1x2) = 8
    # 4 * 11 (2x2x3) = 44
    # 2 * 6 (2x1x3) = 12
    # Total = 28 + 8 + 44 + 12 = 92
    
    global_n, global_m, global_p = n, m, p
    total_rank = 0
    U_entries = []
    V_entries = []
    W_entries = []
    
    def add_block(i0, j0, k0, bn, bm, bp, rank_offset):
        nonlocal U_entries, V_entries, W_entries
        u_e, v_e, w_e = _construct_block(i0, j0, k0, bn, bm, bp, global_n, global_m, global_p)
        for row, col, val in u_e:
            U_entries.append((row, col + rank_offset, val))
        for row, col, val in v_e:
            V_entries.append((row, col + rank_offset, val))
        for row, col, val in w_e:
            W_entries.append((row, col + rank_offset, val))
        return rank_offset
    
    # 2x2x3 decomposition helper
    def add_2x2x3_block(i0, j0, k0, rank_offset):
        # 2x2x2 at k0, k0+1
        r1 = add_block(i0, j0, k0, 2, 2, 2, rank_offset)
        # 2x2x1 at k0+2
        r2 = add_block(i0, j0, k0+2, 2, 2, 1, r1)
        return r2

    # 2x1x3 decomposition helper
    def add_2x1x3_block(i0, j0, k0, rank_offset):
        # 2x1x2 at k0, k0+1
        r1 = add_block(i0, j0, k0, 2, 1, 2, rank_offset)
        # 2x1x1 at k0+2
        r2 = add_block(i0, j0, k0+2, 2, 1, 1, r1)
        return r2

    # Block 1: i=0,1; j=0,1; k=0,1
    total_rank = add_block(0, 0, 0, 2, 2, 2, 0)
    # Block 2: i=0,1; j=2,3; k=0,1
    total_rank = add_block(0, 2, 0, 2, 2, 2, total_rank)
    # Block 3: i=0,1; j=4; k=0,1
    total_rank = add_block(0, 4, 0, 2, 1, 2, total_rank)
    # Block 4: i=2,3; j=0,1; k=0,1
    total_rank = add_block(2, 0, 0, 2, 2, 2, total_rank)
    # Block 5: i=2,3; j=2,3; k=0,1
    total_rank = add_block(2, 2, 0, 2, 2, 2, total_rank)
    # Block 6: i=2,3; j=4; k=0,1
    total_rank = add_block(2, 4, 0, 2, 1, 2, total_rank)
    
    # Block 7: i=0,1; j=0,1; k=2,3,4
    total_rank = add_2x2x3_block(0, 0, 2, total_rank)
    # Block 8: i=0,1; j=2,3; k=2,3,4
    total_rank = add_2x2x3_block(0, 2, 2, total_rank)
    # Block 9: i=0,1; j=4; k=2,3,4
    total_rank = add_2x1x3_block(0, 4, 2, total_rank)
    # Block 10: i=2,3; j=0,1; k=2,3,4
    total_rank = add_2x2x3_block(2, 0, 2, total_rank)
    # Block 11: i=2,3; j=2,3; k=2,3,4
    total_rank = add_2x2x3_block(2, 2, 2, total_rank)
    # Block 12: i=2,3; j=4; k=2,3,4
    total_rank = add_2x1x3_block(2, 4, 2, total_rank)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r, c, val in U_entries:
        U[r][c] += val
    for r, c, val in V_entries:
        V[r][c] += val
    for r, c, val in W_entries:
        W[r][c] += val
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
