"""
Program to find the rank-minimal exact tensor decomposition for multiplying
a 2 by 2 matrix by a 2 by 15 matrix.

Uses Strassen's algorithm for the 2x2x2 blocks (rank 7 each) where possible,
and schoolbook (rank 4) for the remaining 2x2x1 block.

Total rank = 7 * 7 + 4 = 53
"""

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions (2, 2, 15).
    
    Uses Strassen's algorithm for 7 blocks of size (2,2,2) and schoolbook
    for the final block of size (2,2,1).
    
    Rank = 7*7 + 4 = 53
    
    Returns:
        dict: A dictionary containing the keys 'U', 'V', 'W' with the factor matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (30, rank)
    
    # Strassen 7 multiplications for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B00+B10)
    # M7 = (A10-A00)(B11+B01)
    #
    # C00 = M1 - M4 + M5
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M3 + M4
    
    # U vectors for each Strassen multiplication (indexed by A_i_j, i,j in {0,1}):
    # A has rows i=0,1 and cols j=0,1; U index = i*m + j = i*2 + j
    # So: A00=0, A01=1, A10=2, A11=3
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 0, 1, 0],   # M5: A00 + A10
        [0, 1, 0, -1],  # M6: A01 - A11
        [0, 0, 1, -1],  # M7: A10 - A00  wait, that's A10 - A00? No: (A10-A00)(B11+B01)
    ]
    # Wait, let me re-check M7: (A10 - A00)(B11 + B01)
    # So U for M7 = [ -1, 0, 1, 0 ]  (A10 - A00 means -1*A00 + 1*A10)
    strassen_U[6] = [-1, 0, 1, 0]
    
    # V vectors for each Strassen multiplication (indexed by B_j_k, j in {0,1}, k in {0,1} for a block):
    # Within a block, B has cols k=0,1. Global B index = j*p + k
    # For block starting at col offset ko: B_j0 -> j*15 + ko, B_j1 -> j*15 + ko + 1
    # V local indices within block: B00=0, B01=1, B10=2, B11=3
    strassen_V_local = [
        [1, 1, 0, 0],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00 + B10
        [0, 1, 0, 1],   # M7: B01 + B11
    ]
    
    # W vectors for each Strassen multiplication (indexed by C_i_k, i in {0,1}, k in {0,1} for a block):
    # W index = k*n + i = k*2 + i
    # Within a block, C has rows i=0,1 and cols k=0,1.
    # Global W index for block starting at row offset ro: C_ik -> (ro+k)*2 + i
    # Local: C00=0, C01=1, C10=2, C11=3
    strassen_W_local = [
        [1, 0, 0, -1],  # M1: C00 - C11 + ... wait
        # C00 = M1 - M4 + M5 => M1 contributes +1 to C00
        # C01 = M3 + M7 => M1 contributes 0 to C01
        # C10 = M2 + M6 => M1 contributes 0 to C10
        # C11 = M1 - M3 + M4 => M1 contributes +1 to C11
        # So M1: [1, 0, 0, 1]
    ]
    strassen_W_local[0] = [1, 0, 0, 1]
    
    # M2: C10 = M2 + M6 => +1 to C10
    # C00 = M1 - M4 + M5 => 0
    # C01 = M3 + M7 => 0
    # C11 = M1 - M3 + M4 => 0
    strassen_W_local[1] = [0, 0, 1, 0]
    
    # M3: C01 = M3 + M7 => +1 to C01; C11 = M1 - M3 + M4 => -1 to C11
    strassen_W_local[2] = [0, 1, 0, -1]
    
    # M4: C00 = M1 - M4 + M5 => -1 to C00; C11 = M1 - M3 + M4 => +1 to C11
    strassen_W_local[3] = [-1, 0, 0, 1]
    
    # M5: C00 = M1 - M4 + M5 => +1 to C00
    strassen_W_local[4] = [1, 0, 0, 0]
    
    # M6: C10 = M2 + M6 => +1 to C10
    strassen_W_local[5] = [0, 0, 1, 0]
    
    # M7: C01 = M3 + M7 => +1 to C01
    strassen_W_local[6] = [0, 1, 0, 0]
    
    # Now build the full decomposition
    # We have 15 columns in B (and 15 rows in C since p=15, n=2, so W has 30 rows)
    # B is 2x15, so B has 2 rows and 15 columns
    # C is 2x15, so C has 2 rows and 15 columns
    
    # We can group the 15 columns into:
    # 7 blocks of 2 columns (columns 0-13) + 1 block of 1 column (column 14)
    # Each block of 2 columns gives a 2x2x2 subproblem (A is 2x2, B_block is 2x2, C_block is 2x2)
    # The last block gives 2x2x1 (A is 2x2, B_block is 2x1, C_block is 2x1)
    
    rank = 7 * 7 + 4  # 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    col_offset = 0
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for block_idx in range(7):
        ko = 2 * block_idx  # starting column offset in B
        ro = 2 * block_idx  # starting row offset in C (since C has same structure)
        
        for mult_idx in range(7):
            r = col_offset + mult_idx
            
            # Fill U
            for a_idx in range(n * m):
                U[a_idx][r] = strassen_U[mult_idx][a_idx]
            
            # Fill V: local indices B00, B01, B10, B11 map to global:
            # B_j_k where j in {0,1}, k in {0,1}
            # Local index 0 -> B00 -> j=0, k=ko -> global row = 0*15 + ko = ko
            # Local index 1 -> B01 -> j=0, k=ko+1 -> global row = 0*15 + ko+1 = ko+1
            # Local index 2 -> B10 -> j=1, k=ko -> global row = 1*15 + ko = 15+ko
            # Local index 3 -> B11 -> j=1, k=ko+1 -> global row = 1*15 + ko+1 = 15+ko+1
            v_global_indices = [ko, ko + 1, 15 + ko, 15 + ko + 1]
            for local_idx in range(4):
                V[v_global_indices[local_idx]][r] = strassen_V_local[mult_idx][local_idx]
            
            # Fill W: local indices C00, C01, C10, C11 map to global:
            # C_i_k where i in {0,1}, k in {0,1}
            # W index = k*n + i = k*2 + i
            # Local index 0 -> C00 -> k=0, i=0 -> global row = ro*2 + 0 = 2*ro
            # Local index 1 -> C01 -> k=1, i=0 -> global row = (ro+1)*2 + 0 = 2*ro + 2
            # Local index 2 -> C10 -> k=0, i=1 -> global row = ro*2 + 1 = 2*ro + 1
            # Local index 3 -> C11 -> k=1, i=1 -> global row = (ro+1)*2 + 1 = 2*ro + 3
            w_global_indices = [2*ro, 2*ro+2, 2*ro+1, 2*ro+3]
            for local_idx in range(4):
                W[w_global_indices[local_idx]][r] = strassen_W_local[mult_idx][local_idx]
    
    col_offset = 7 * 7  # 49
    
    # Last block: 2x2x1 (column 14 of B, row 14 of C)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 4 terms:
    # Term 0: A00*B0 -> C0
    # Term 1: A01*B1 -> C0
    # Term 2: A10*B0 -> C1
    # Term 3: A11*B1 -> C1
    
    ko = 14  # column offset
    ro = 14  # row offset
    
    # V global indices: B_j_k where k=14 only
    # B0_14 -> j=0, k=14 -> global row = 0*15 + 14 = 14
    # B1_14 -> j=1, k=14 -> global row = 1*15 + 14 = 29
    # W global indices: C_i_k where k=14
    # C0_14 -> k=14, i=0 -> global row = 14*2 + 0 = 28
    # C1_14 -> k=14, i=1 -> global row = 14*2 + 1 = 29
    
    # Term 0: A00 * B0_14 -> C0_14
    r = col_offset
    U[0][r] = 1  # A00
    V[14][r] = 1  # B0_14
    W[28][r] = 1  # C0_14
    
    # Term 1: A01 * B1_14 -> C0_14
    r = col_offset + 1
    U[1][r] = 1  # A01
    V[29][r] = 1  # B1_14
    W[28][r] = 1  # C0_14
    
    # Term 2: A10 * B0_14 -> C1_14
    r = col_offset + 2
    U[2][r] = 1  # A10
    V[14][r] = 1  # B0_14
    W[29][r] = 1  # C1_14
    
    # Term 3: A11 * B1_14 -> C1_14
    r = col_offset + 3
    U[3][r] = 1  # A11
    V[29][r] = 1  # B1_14
    W[29][r] = 1  # C1_14
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
