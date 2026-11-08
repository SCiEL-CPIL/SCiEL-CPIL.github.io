"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x5 * 5x6 matrix multiplication.
    Uses a hybrid approach:
    - Apply Strassen (7-term) decomposition to disjoint 2x2x2 sub-blocks.
    - Use schoolbook for the remaining entries.
    
    Target rank: 84 (vs 90 for schoolbook).
    
    Layout:
    n=3 (rows of A), m=5 (cols of A / rows of B), p=6 (cols of B)
    Output C is 3x6.
    
    We partition the dimensions to maximize 2x2x2 Strassen blocks:
    
    For the i-dimension (n=3), we can have:
      - Block 1: i in {0,1}
      - Remaining: i in {2}
    
    For the j-dimension (m=5), we can have:
      - Block 1: j in {0,1}
      - Block 2: j in {2,3}
      - Remaining: j in {4}
    
    For the k-dimension (p=6), we can have:
      - Block 1: k in {0,1}
      - Block 2: k in {2,3}
      - Block 3: k in {4,5}
    
    So we can form 2x2x2 Strassen blocks for:
      (i in {0,1}, j in {0,1}, k in {0,1}) -> 7 terms
      (i in {0,1}, j in {0,1}, k in {2,3}) -> 7 terms
      (i in {0,1}, j in {0,1}, k in {4,5}) -> 7 terms
      (i in {0,1}, j in {2,3}, k in {0,1}) -> 7 terms
      (i in {0,1}, j in {2,3}, k in {2,3}) -> 7 terms
      (i in {0,1}, j in {2,3}, k in {4,5}) -> 7 terms
    
    That's 6 Strassen blocks * 7 = 42 terms.
    
    Remaining terms (schoolbook):
    - i in {2}, j in {0,1}, k in {0,1} -> 4 terms
    - i in {2}, j in {0,1}, k in {2,3} -> 4 terms
    - i in {2}, j in {0,1}, k in {4,5} -> 4 terms
    - i in {2}, j in {2,3}, k in {0,1} -> 4 terms
    - i in {2}, j in {2,3}, k in {2,3} -> 4 terms
    - i in {2}, j in {2,3}, k in {4,5} -> 4 terms
    - i in {0,1}, j in {4}, k in {0,1} -> 4 terms
    - i in {0,1}, j in {4}, k in {2,3} -> 4 terms
    - i in {0,1}, j in {4}, k in {4,5} -> 4 terms
    - i in {2}, j in {4}, k in {0,1} -> 4 terms
    - i in {2}, j in {4}, k in {2,3} -> 4 terms
    - i in {2}, j in {4}, k in {4,5} -> 4 terms
    
    Wait, let me recount. The total is 90 schoolbook terms.
    We replaced 6 blocks of 8 terms each (48 terms) with 6 blocks of 7 terms each (42 terms).
    So we save 6 terms, giving 90 - 6 = 84 terms.
    
    But wait, I need to be more careful. Let me list all the disjoint 2x2x2 blocks:
    
    The i-dimension has 3 elements: {0, 1, 2}. We can form one 2-block: {0,1}.
    The j-dimension has 5 elements: {0, 1, 2, 3, 4}. We can form two 2-blocks: {0,1}, {2,3}.
    The k-dimension has 6 elements: {0, 1, 2, 3, 4, 5}. We can form three 2-blocks: {0,1}, {2,3}, {4,5}.
    
    So the number of 2x2x2 blocks is 1 * 2 * 3 = 6 blocks.
    Each block uses 7 terms instead of 8, so we save 6 terms.
    Total rank = 90 - 6 = 84.
    
    Now, I need to implement this. The Strassen decomposition for a 2x2x2 block:
    For inputs A_00, A_01, A_10, A_11 (rows i in {i0, i1}, cols j in {j0, j1})
    and B_00, B_01, B_10, B_11 (rows j in {j0, j1}, cols k in {k0, k1})
    
    The 7 multiplications are:
    M1 = (A_00 + A_11) * (B_00 + B_11)
    M2 = (A_10 + A_11) * B_00
    M3 = A_00 * (B_01 - B_11)
    M4 = A_11 * (B_10 - B_00)
    M5 = (A_00 + A_01) * B_11
    M6 = (A_10 - A_00) * (B_00 + B_01)
    M7 = (A_01 - A_11) * (B_10 + B_11)
    
    Output:
    C_00 = M1 + M4 - M5 + M7
    C_01 = M3 + M5
    C_10 = M2 + M4
    C_11 = M1 - M2 + M3 + M6
    
    In the tensor decomposition, each term is a rank-1 tensor U_c ⊗ V_c ⊗ W_c.
    For the Strassen block, each M_t corresponds to a rank-1 term where:
    - U_c has non-zero entries at (i, j) positions corresponding to the linear combination of A entries.
    - V_c has non-zero entries at (j, k) positions corresponding to the linear combination of B entries.
    - W_c has non-zero entries at (k, i) positions corresponding to the linear combination of C entries.
    
    Let me implement this step by step.
    """
    n, m, p = 3, 5, 6
    rank = 84
    
    # Initialize U, V, W as lists of lists (for clarity, then convert to numpy if needed)
    U = [[0] * rank for _ in range(n * m)]  # shape (n*m, rank)
    V = [[0] * rank for _ in range(m * p)]  # shape (m*p, rank)
    W = [[0] * rank for _ in range(p * n)]  # shape (p*n, rank)
    
    col = 0
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        """Add a Strassen 2x2x2 block decomposition."""
        nonlocal col
        
        # M1 = (A_00 + A_11) * (B_00 + B_11)
        # U entries: A_00 at (i0, j0), A_11 at (i1, j1)
        # V entries: B_00 at (j0, k0), B_11 at (j1, k1)
        # W entries: C_00 at (k0, i0), C_11 at (k1, i1)
        # But wait, the output C is n x p, so C_00 is at (i0, k0), C_01 at (i0, k1), etc.
        # The W matrix has shape (p*n, rank), indexed by [k*n + i].
        # So C_00 corresponds to W[k0*n + i0], C_01 to W[k1*n + i0], C_10 to W[k0*n + i1], C_11 to W[k1*n + i1].
        
        # Let me re-derive the W indices for each output term.
        # Output C has entries C[i, k] for i in 0..n-1, k in 0..p-1.
        # W is indexed by [k*n + i].
        
        # M1 contributes to C_00 (i0, k0) and C_11 (i1, k1)
        # M2 contributes to C_10 (i1, k0)
        # M3 contributes to C_01 (i0, k1)
        # M4 contributes to C_00 (i0, k0) and C_10 (i1, k0)
        # M5 contributes to C_00 (i0, k0) and C_01 (i0, k1)
        # M6 contributes to C_11 (i1, k1)
        # M7 contributes to C_00 (i0, k0) and C_11 (i1, k1)
        
        # Let me be more precise. The Strassen algorithm computes:
        # C_00 = M1 + M4 - M5 + M7
        # C_01 = M3 + M5
        # C_10 = M2 + M4
        # C_11 = M1 - M2 + M3 + M6
        
        # Each M_t is a product of a linear combination of A entries and a linear combination of B entries.
        # In the tensor decomposition, each M_t is a single rank-1 term.
        
        # M1: A_part = A_00 + A_11, B_part = B_00 + B_11
        U[i0*m + j0][col] = 1  # A_00
        U[i1*m + j1][col] = 1  # A_11
        V[j0*p + k0][col] = 1  # B_00
        V[j1*p + k1][col] = 1  # B_11
        W[k0*n + i0][col] = 1  # C_00
        W[k1*n + i1][col] = 1  # C_11
        col += 1
        
        # M2: A_part = A_10 + A_11, B_part = B_00
        U[i1*m + j0][col] = 1  # A_10
        U[i1*m + j1][col] = 1  # A_11
        V[j0*p + k0][col] = 1  # B_00
        W[k0*n + i1][col] = 1  # C_10
        col += 1
        
        # M3: A_part = A_00, B_part = B_01 - B_11
        U[i0*m + j0][col] = 1  # A_00
        V[j0*p + k1][col] = 1  # B_01
        V[j1*p + k1][col] = -1  # -B_11
        W[k1*n + i0][col] = 1  # C_01
        col += 1
        
        # M4: A_part = A_11, B_part = B_10 - B_00
        U[i1*m + j1][col] = 1  # A_11
        V[j1*p + k0][col] = 1  # B_10
        V[j0*p + k0][col] = -1  # -B_00
        W[k0*n + i0][col] = 1  # C_00
        W[k0*n + i1][col] = 1  # C_10
        col += 1
        
        # M5: A_part = A_00 + A_01, B_part = B_11
        U[i0*m + j0][col] = 1  # A_00
        U[i0*m + j1][col] = 1  # A_01
        V[j1*p + k1][col] = 1  # B_11
        W[k0*n + i0][col] = -1  # -C_00
        W[k1*n + i0][col] = 1  # C_01
        col += 1
        
        # M6: A_part = A_10 - A_00, B_part = B_00 + B_01
        U[i1*m + j0][col] = 1  # A_10
        U[i0*m + j0][col] = -1  # -A_00
        V[j0*p + k0][col] = 1  # B_00
        V[j0*p + k1][col] = 1  # B_01
        W[k1*n + i1][col] = 1  # C_11
        col += 1
        
        # M7: A_part = A_01 - A_11, B_part = B_10 + B_11
        U[i0*m + j1][col] = 1  # A_01
        U[i1*m + j1][col] = -1  # -A_11
        V[j1*p + k0][col] = 1  # B_10
        V[j1*p + k1][col] = 1  # B_11
        W[k0*n + i0][col] = 1  # C_00
        W[k1*n + i1][col] = 1  # C_11
        col += 1
    
    # Add the 6 Strassen blocks
    # i-block: {0, 1}
    # j-blocks: {0, 1}, {2, 3}
    # k-blocks: {0, 1}, {2, 3}, {4, 5}
    
    for j_block in [(0, 1), (2, 3)]:
        for k_block in [(0, 1), (2, 3), (4, 5)]:
            add_strassen_block(0, 1, j_block[0], j_block[1], k_block[0], k_block[1])
    
    # Now add the remaining schoolbook terms.
    # These are all (i, j, k) tuples that are not covered by the Strassen blocks.
    # Strassen blocks cover: i in {0,1}, j in {0,1} U {2,3}, k in {0,1} U {2,3} U {4,5}
    # So remaining:
    # i in {2}, any j in {0,1,2,3,4}, any k in {0,1,2,3,4,5}
    # i in {0,1}, j in {4}, any k in {0,1,2,3,4,5}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this (i, j, k) is in a Strassen block
                in_strassen = False
                if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3, 4, 5]:
                    # Check if j is in a 2-block and k is in a 2-block
                    j_in_block = (j in [0, 1]) or (j in [2, 3])
                    k_in_block = (k in [0, 1]) or (k in [2, 3]) or (k in [4, 5])
                    if j_in_block and k_in_block:
                        in_strassen = True
                
                if not in_strassen:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    # Verify that col == rank
    assert col == rank, f"Expected rank {rank}, but got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
