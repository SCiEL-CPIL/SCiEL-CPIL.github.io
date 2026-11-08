"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-56 decomposition for 2x4x8 matrix multiplication.
    This uses 4 Strassen decompositions (rank 7 each) for each 2x4x4 block,
    and there are two such blocks for the 8 columns.
    Total rank = 2 * 4 * 7 = 56.
    """
    n, m, p = DIMENSIONS
    
    # We will build U, V, W for the full 2x4x8 problem.
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    # The block corresponds to:
    # A rows: i1, i2 (global i indices)
    # B cols: j1, j2 (global j indices)
    # C cols: k1, k2 (global k indices)
    # Note: The problem is C = A * B.
    # A is n x m, B is m x p, C is n x p.
    # Indices: A[i,j], B[j,k], C[i,k].
    # W index convention: W[k*n + i, r]
    
    def add_strassen(i_base, j_base, k_base, col_offset):
        # i_base: 0 or 1 (offset in n)
        # j_base: 0, 1, 2, 3 (offset in m)
        # k_base: 0, 1, 2, 3 (offset in p)
        
        # The 2x2 blocks are:
        # A11 = A[i_base, j_base], A12 = A[i_base, j_base+1]
        # A21 = A[i_base+1, j_base], A22 = A[i_base+1, j_base+1]
        # B11 = B[j_base, k_base], B12 = B[j_base, k_base+1]
        # B21 = B[j_base+1, k_base], B22 = B[j_base+1, k_base+1]
        
        # Strassen intermediates:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to U, V, W.
        # U entries are for A indices (i*m + j).
        # V entries are for B indices (j*p + k).
        # W entries are for C indices (k*n + i).
        
        # M1
        U[i_base * m + j_base][col_offset + 0] += 1
        U[(i_base + 1) * m + (j_base + 1)][col_offset + 0] += 1
        V[j_base * p + k_base][col_offset + 0] += 1
        V[(j_base + 1) * p + (k_base + 1)][col_offset + 0] += 1
        # C11 += M1, C22 += M1
        W[k_base * n + i_base][col_offset + 0] += 1
        W[(k_base + 1) * n + (i_base + 1)][col_offset + 0] += 1
        
        # M2
        U[(i_base + 1) * m + j_base][col_offset + 1] += 1
        U[(i_base + 1) * m + (j_base + 1)][col_offset + 1] += 1
        V[j_base * p + k_base][col_offset + 1] += 1
        # C12 += M2? No, C12 = A11 B12 + A12 B22.
        # Strassen: C11 = M1 - M4 + M6, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M3 + M7
        # Wait, standard Strassen:
        # C11 = M1 - M4 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M7
        
        # So for M2:
        # A part: A21 + A22
        # B part: B11
        # C part: C21 += M2
        W[k_base * n + (i_base + 1)][col_offset + 1] += 1
        
        # M3
        # A part: A11
        # B part: B12 - B22
        U[i_base * m + j_base][col_offset + 2] += 1
        V[j_base * p + (k_base + 1)][col_offset + 2] += 1
        V[(j_base + 1) * p + (k_base + 1)][col_offset + 2] -= 1
        # C part: C12 += M3
        W[(k_base + 1) * n + i_base][col_offset + 2] += 1
        
        # M4
        # A part: A22
        # B part: B21 - B11
        U[(i_base + 1) * m + (j_base + 1)][col_offset + 3] += 1
        V[(j_base + 1) * p + k_base][col_offset + 3] += 1
        V[j_base * p + k_base][col_offset + 3] -= 1
        # C part: C11 -= M4, C21 += M4
        W[k_base * n + i_base][col_offset + 3] -= 1
        W[k_base * n + (i_base + 1)][col_offset + 3] += 1
        
        # M5
        # A part: A11 + A12
        # B part: B22
        U[i_base * m + j_base][col_offset + 4] += 1
        U[i_base * m + (j_base + 1)][col_offset + 4] += 1
        V[(j_base + 1) * p + (k_base + 1)][col_offset + 4] += 1
        # C part: C12 += M5
        W[(k_base + 1) * n + i_base][col_offset + 4] += 1
        
        # M6
        # A part: A21 - A11
        # B part: B11 + B12
        U[(i_base + 1) * m + j_base][col_offset + 5] += 1
        U[i_base * m + j_base][col_offset + 5] -= 1
        V[j_base * p + k_base][col_offset + 5] += 1
        V[j_base * p + (k_base + 1)][col_offset + 5] += 1
        # C part: C11 += M6
        W[k_base * n + i_base][col_offset + 5] += 1
        
        # M7
        # A part: A12 - A22
        # B part: B21 + B22
        U[i_base * m + (j_base + 1)][col_offset + 6] += 1
        U[(i_base + 1) * m + (j_base + 1)][col_offset + 6] -= 1
        V[(j_base + 1) * p + k_base][col_offset + 6] += 1
        V[(j_base + 1) * p + (k_base + 1)][col_offset + 6] += 1
        # C part: C22 += M7
        W[(k_base + 1) * n + (i_base + 1)][col_offset + 6] += 1

    # Iterate over the 2x4x8 structure
    # n=2, m=4, p=8
    # We split p into two halves of 4.
    # For each half, we split m into two halves of 2.
    # So we have 2 (p-halves) * 2 (m-halves) = 4 blocks of 2x2x2.
    
    col_offset = 0
    for k_half in range(2):  # 0, 1
        k_base = k_half * 4
        for j_half in range(2):  # 0, 1
            j_base = j_half * 2
            # i_base is always 0 because n=2
            add_strassen(0, j_base, k_base, col_offset)
            col_offset += 7
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
