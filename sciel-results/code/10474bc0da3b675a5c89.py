"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    # Block 1: k=0,1 (2x3x2) -> Rank 11
    # Block 2: k=2,3 (2x3x2) -> Rank 11
    # Block 3: k=4   (2x3x1) -> Rank 6
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # a_offset: index offset for A rows (i*2 + i_local)
    # b_offset: index offset for B cols (j*5 + k_local) - wait, B is 3x5.
    # In our split, A is 2x2 submatrix, B is 2x2 submatrix.
    # We need to map local indices to global.
    
    def add_strassen(a_i_start, b_j_start, c_k_start):
        nonlocal col
        # a_i_start: global i for the 2x2 A block (0 or 1? No, A is 2x3. We use rows 0,1 of A. So i is 0,1 always for A rows? 
        # Wait, A is 2x3. The 2x2 block uses A[0:2, j1:j1+2].
        # So global i is 0,1.
        # Global j for A is j1, j1+1.
        # Global j for B is j1, j1+1.
        # Global k for C is k1, k1+1.
        
        # Strassen terms M1..M7
        # M1: (A00+A11)(B00+B11)
        U[a_i_start * m + 0][col] += 1 # A[0,0]
        U[a_i_start * m + 1][col] += 1 # A[1,0] -- Wait, A is 2x3.
        # Let's be precise.
        # A global row i, col j.
        # In 2x2 block, local i, local j.
        # Global i = i (since A has 2 rows, i is 0 or 1).
        # Global j = j_start + local_j.
        
        # Let's re-define the mapping carefully.
        # A is 2x3. We select columns j1, j2.
        # B is 3x5. We select rows j1, j2 and columns k1, k2.
        # C is 2x5. We select rows i (0,1) and columns k1, k2.
        
        pass # Logic implemented below directly to avoid closure complexity in snippet

    # Block 1: k=0,1. A cols 0,1,2. B rows 0,1,2.
    # Split A into A1 (col 0) and A2 (cols 1,2).
    # Split B1 (cols 0,1 of B) into B11 (row 0) and B12 (rows 1,2).
    
    # Part 1a: A1 * B11 (2x1 * 1x2) -> 2x2. Rank 4.
    # A1 is A[:, 0]. B11 is B[0, 0:2].
    # C_part = A1 * B11.
    # C[i, k] = A[i, 0] * B[0, k].
    for i in range(2):
        for k_local in range(2):
            # U[i, 0] = 1
            # V[0, k_global] = 1
            # W[k_global, i] = 1
            U[i * m + 0][col] = 1
            V[0 * p + k_local][col] = 1
            W[k_local * n + i][col] = 1
            col += 1
            
    # Part 1b: A2 * B12 (2x2 * 2x2) -> 2x2. Rank 7.
    # A2 is A[:, 1:3]. B12 is B[1:3, 0:2].
    # Strassen on A2, B12.
    # Local A indices: i in 0,1. j in 0,1 (global j = 1+j_local).
    # Local B indices: j in 0,1 (global j = 1+j_local). k in 0,1 (global k = k_local).
    
    def add_strassen_block(a_j_start, b_j_start, c_k_start):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[0 * m + a_j_start][col] += 1
        U[1 * m + a_j_start][col] += 1
        U[0 * m + a_j_start + 1][col] += 1
        U[1 * m + a_j_start + 1][col] += 1
        
        V[b_j_start * p + c_k_start][col] += 1
        V[b_j_start * p + c_k_start + 1][col] += 1
        V[b_j_start + 1 * p + c_k_start][col] += 1
        V[b_j_start + 1 * p + c_k_start + 1][col] += 1
        
        W[c_k_start * n + 0][col] += 1
        W[c_k_start * n + 1][col] += 1
        W[c_k_start + 1 * n + 0][col] += 1
        W[c_k_start + 1 * n + 1][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[1 * m + a_j_start][col] += 1
        U[1 * m + a_j_start + 1][col] += 1
        V[b_j_start * p + c_k_start][col] += 1
        W[c_k_start * n + 1][col] += 1
        W[c_k_start + 1 * n + 1][col] += 1
        col += 1
        
        # M3: A00(B10-B11)
        U[0 * m + a_j_start][col] += 1
        V[b_j_start + 1 * p + c_k_start][col] += 1
        V[b_j_start + 1 * p + c_k_start + 1][col] -= 1
        W[c_k_start * n + 0][col] += 1
        W[c_k_start + 1 * n + 0][col] += 1
        col += 1
        
        # M4: A11(B01-B00)
        U[1 * m + a_j_start + 1][col] += 1
        V[b_j_start * p + c_k_start + 1][col] += 1
        V[b_j_start * p + c_k_start][col] -= 1
        W[c_k_start * n + 1][col] += 1
        W[c_k_start + 1 * n + 1][col] += 1
        col += 1
        
        # M5: (A00+A10)B11
        U[0 * m + a_j_start][col] += 1
        U[1 * m + a_j_start][col] += 1
        V[b_j_start + 1 * p + c_k_start + 1][col] += 1
        W[c_k_start * n + 0][col] += 1
        W[c_k_start + 1 * n + 0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1 * m + a_j_start][col] += 1
        U[0 * m + a_j_start][col] -= 1
        V[b_j_start * p + c_k_start][col] += 1
        V[b_j_start * p + c_k_start + 1][col] += 1
        W[c_k_start * n + 1][col] += 1
        W[c_k_start + 1 * n + 1][col] += 1
        col += 1
        
        # M7: (A11-A01)(B10+B11)
        U[1 * m + a_j_start + 1][col] += 1
        U[0 * m + a_j_start + 1][col] -= 1
        V[b_j_start + 1 * p + c_k_start][col] += 1
        V[b_j_start + 1 * p + c_k_start + 1][col] += 1
        W[c_k_start * n + 0][col] += 1
        W[c_k_start + 1 * n + 0][col] += 1
        col += 1

    # Execute Block 1 Strassen (A cols 1,2; B rows 1,2; C cols 0,1)
    add_strassen_block(1, 1, 0)
    
    # Block 2: k=2,3.
    # Part 2a: A1 * B21 (2x1 * 1x2). B21 is B[0, 2:4].
    for i in range(2):
        for k_local in range(2):
            U[i * m + 0][col] = 1
            V[0 * p + 2 + k_local][col] = 1
            W[(2 + k_local) * n + i][col] = 1
            col += 1
            
    # Part 2b: A2 * B22 (2x2 * 2x2). B22 is B[1:3, 2:4].
    add_strassen_block(1, 1, 2)
    
    # Block 3: k=4.
    # A (2x3) * B3 (3x1). Rank 6.
    for i in range(2):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
