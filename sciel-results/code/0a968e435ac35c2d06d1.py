import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 Strassen for 2x2x2
    # A, B, C are 2x2
    # U_strassen: 4x7
    # V_strassen: 4x7
    # W_strassen: 4x7
    
    # Standard Strassen factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U_strassen rows: A00, A01, A10, A11
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    U_s = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 1, 0, 0, 1, 0, -1], # A01
        [0, 0, 1, 0, 0, 1, 0], # A10
        [1, 0, 1, 1, 0, 0, -1]  # A11
    ]
    
    # V_strassen rows: B00, B01, B10, B11
    V_s = [
        [1, 1, 1, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0], # B01
        [1, 0, -1, 0, 0, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ]
    
    # W_strassen rows: C00, C01, C10, C11
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M5 - M3
    W_s = [
        [1, 0, 0, -1, 0, 1, 1], # C00
        [0, 0, 0, 1, 1, 0, 0], # C01
        [0, 1, 1, 0, 0, 0, 0], # C10
        [1, -1, -1, 0, 1, 0, 0] # C11
    ]
    
    # Tail 2x1x2 Rank 4
    # A is col 3 (indices 0,1 in local 2x1)
    # B is row 3 (indices 0,1 in local 1x2)
    # C is 2x2
    # C00 = A0 B0
    # C01 = A0 B1
    # C10 = A1 B0
    # C11 = A1 B1
    
    U_t = [
        [1, 1, 0, 0], # A0
        [0, 0, 1, 1]  # A1
    ]
    V_t = [
        [1, 0, 1, 0], # B0
        [0, 1, 0, 1]  # B1
    ]
    W_t = [
        [1, 0, 0, 0], # C00
        [0, 1, 0, 0], # C01
        [0, 0, 1, 0], # C10
        [0, 0, 0, 1]  # C11
    ]
    
    rank_total = 33
    U = [[0] * rank_total for _ in range(n * m)] # 6 x 33
    V = [[0] * rank_total for _ in range(m * p)] # 18 x 33
    W = [[0] * rank_total for _ in range(p * n)] # 12 x 33
    
    col = 0
    for block in range(3):
        # Strassen part
        for r in range(7):
            # U: A indices 0,1,2,3 (cols 0,1 of A)
            # A index in U matrix: i*m + j. i=0,1; j=0,1.
            # So 0,1,2,3.
            U[0][col] += U_s[0][r]
            U[1][col] += U_s[1][r]
            U[2][col] += U_s[2][r]
            U[3][col] += U_s[3][r]
            
            # V: B indices for rows 0,1 of block.
            # B index in V matrix: j*p + k.
            # j=0,1; k=0,1 (local).
            # Block start k = block*2.
            # So j*p + (block*2 + k_local).
            # V_s indices 0,1,2,3 correspond to B00, B01, B10, B11.
            # B00: j=0, k_local=0 -> 0*6 + block*2
            # B01: j=0, k_local=1 -> 0*6 + block*2 + 1
            # B10: j=1, k_local=0 -> 1*6 + block*2
            # B11: j=1, k_local=1 -> 1*6 + block*2 + 1
            V[0*6 + block*2][col] += V_s[0][r]
            V[0*6 + block*2 + 1][col] += V_s[1][r]
            V[1*6 + block*2][col] += V_s[2][r]
            V[1*6 + block*2 + 1][col] += V_s[3][r]
            
            # W: C indices for block.
            # W index: k*n + i.
            # k_local=0,1; i=0,1.
            # k = block*2 + k_local.
            # W_s indices 0,1,2,3 correspond to C00, C01, C10, C11.
            # C00: i=0, k_local=0 -> (block*2)*2 + 0
            # C01: i=1, k_local=0 -> (block*2)*2 + 1 ? No.
            # W[k*n + i].
            # C_{ik}. k is col of C (local), i is row of C.
            # W_s[0] is C00 (i=0, k_local=0). Index: (block*2 + 0)*2 + 0.
            # W_s[1] is C01 (i=0, k_local=1). Index: (block*2 + 1)*2 + 0.
            # W_s[2] is C10 (i=1, k_local=0). Index: (block*2 + 0)*2 + 1.
            # W_s[3] is C11 (i=1, k_local=1). Index: (block*2 + 1)*2 + 1.
            
            W[(block*2 + 0)*2 + 0][col] += W_s[0][r]
            W[(block*2 + 1)*2 + 0][col] += W_s[1][r]
            W[(block*2 + 0)*2 + 1][col] += W_s[2][r]
            W[(block*2 + 1)*2 + 1][col] += W_s[3][r]
            
            col += 1
            
        # Tail part
        for r in range(4):
            # U: A index for col 2 (local col 3).
            # A index i*m + j. j=2.
            # A02: i=0, j=2 -> 0*3 + 2 = 2.
            # A12: i=1, j=2 -> 1*3 + 2 = 5.
            # Wait, U_s used 0,1,2,3 for j=0,1.
            # U_t uses A0, A1 which correspond to A02, A12.
            U[2][col] += U_t[0][r]
            U[5][col] += U_t[1][r]
            
            # V: B index for row 2 (local row 3).
            # B index j*p + k. j=2.
            # B2, k_local=0 -> 2*6 + block*2
            # B2, k_local=1 -> 2*6 + block*2 + 1
            V[2*6 + block*2][col] += V_t[0][r]
            V[2*6 + block*2 + 1][col] += V_t[1][r]
            
            # W: C indices for block.
            # Same mapping as Strassen.
            W[(block*2 + 0)*2 + 0][col] += W_t[0][r]
            W[(block*2 + 1)*2 + 0][col] += W_t[1][r]
            W[(block*2 + 0)*2 + 1][col] += W_t[2][r]
            W[(block*2 + 1)*2 + 1][col] += W_t[3][r]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
