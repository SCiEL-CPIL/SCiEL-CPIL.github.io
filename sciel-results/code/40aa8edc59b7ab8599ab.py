import numpy as np

def construct():
    # Rank 28 decomposition for 2x4x4 matrix multiplication
    # Strategy:
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x4) into B1 (2x4) and B2 (2x4)
    # C1 = A1 * B1 (2x4)
    # C2 = A2 * B2 (2x4)
    # C = [C1, C2]
    # Rank(M_{2,2,4}) = 14 (two Strassen instances sharing B)
    # Total Rank = 14 + 14 = 28
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Helper: Get Strassen factors for M_{2,2,2}
    # Returns U_sub (4x7), V_sub (4x7), W_sub (4x7)
    def get_strassen_222():
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # M1
        U_s[0][0] = 1; U_s[3][0] = 1 # A11, A22
        V_s[0][0] = 1; V_s[3][0] = 1 # B11, B22 (indices: B[j*p+k], p=2, so B11=0, B22=3)
        W_s[0][0] = 1; W_s[3][0] = 1 # C11, C22 (indices: C[k*n+i], n=2, so C11=0, C22=3)
        
        # M2
        U_s[2][1] = 1; U_s[3][1] = 1 # A21, A22
        V_s[0][1] = 1 # B11
        W_s[2][1] = 1; W_s[3][1] = 1 # C21, C22
        
        # M3
        U_s[0][2] = 1 # A11
        V_s[1][2] = 1; V_s[3][2] = -1 # B12, B22
        W_s[1][2] = 1; W_s[3][2] = 1 # C12, C22
        
        # M4
        U_s[3][3] = 1 # A22
        V_s[2][3] = 1; V_s[0][3] = -1 # B21, B11
        W_s[0][3] = -1; W_s[2][3] = 1 # C11, C21
        
        # M5
        U_s[0][4] = 1; U_s[1][4] = 1 # A11, A12
        V_s[3][4] = 1 # B22
        W_s[0][4] = 1; W_s[1][4] = 1 # C11, C12
        
        # M6
        U_s[2][5] = 1; U_s[0][5] = -1 # A21, A11
        V_s[0][5] = 1; V_s[1][5] = 1 # B11, B12
        W_s[3][5] = 1 # C22
        
        # M7
        U_s[1][6] = 1; U_s[3][6] = -1 # A12, A22
        V_s[2][6] = 1; V_s[3][6] = 1 # B21, B22
        W_s[0][6] = -1 # C11
        
        return U_s, V_s, W_s

    U_s, V_s, W_s = get_strassen_222()
    
    # Construct M_{2,2,4} decomposition
    # A is 2x2 (indices 0..3 in U)
    # B is 2x4 (indices 0..7 in V) -> B1 (0..3), B2 (4..7)
    # C is 2x4 (indices 0..7 in W) -> C1 (0..3), C2 (4..7)
    # We compute C1 = A * B1 and C2 = A * B2
    # This requires two Strassen instances.
    # Instance 1: A * B1 -> C1
    # Instance 2: A * B2 -> C2
    
    U_224 = [[0]*14 for _ in range(4)]
    V_224 = [[0]*14 for _ in range(8)]
    W_224 = [[0]*14 for _ in range(8)]
    
    # Instance 1: cols 0..6 of U_224, V_224, W_224
    for i in range(4):
        for c in range(7):
            U_224[i][c] = U_s[i][c]
            V_224[i][c] = V_s[i][c] # B1 is first 4 entries of B
            W_224[i][c] = W_s[i][c] # C1 is first 4 entries of C
            
    # Instance 2: cols 7..13 of U_224, V_224, W_224
    for i in range(4):
        for c in range(7):
            U_224[i][c+7] = U_s[i][c]
            V_224[i][c+7] = V_s[i][c] # B2 is next 4 entries of B (offset 4)
            # Wait, V_s indices are 0..3 for B11..B22.
            # For B2, the local indices 0..3 correspond to global 4..7.
            # So V_224[i][c+7] should map V_s[i][c] to global index i+4?
            # No, V_s[i][c] is 1 at local index i.
            # We need to set V_224[local_i + 4][c+7] = V_s[local_i][c].
            pass
            
    # Redo Instance 2 mapping carefully
    for i in range(4):
        for c in range(7):
            if V_s[i][c] != 0:
                V_224[i + 4][c + 7] = V_s[i][c]
            if W_s[i][c] != 0:
                W_224[i + 4][c + 7] = W_s[i][c]
            U_224[i][c + 7] = U_s[i][c]

    # Now assemble M_{2,4,4}
    # A is 2x4. A1 is cols 0..3, A2 is cols 4..7.
    # B is 4x4. B1 is rows 0..3, B2 is rows 4..7.
    # C is 2x4. C1 is cols 0..3, C2 is cols 4..7.
    # C1 = A1 * B1. Uses U_224 rows 0..3, V_224 rows 0..7, W_224 rows 0..7.
    # C2 = A2 * B2. Uses U_224 rows 0..3, V_224 rows 0..7, W_224 rows 0..7.
    # But A2 is at global A indices 4..7. B2 is at global B indices 4..7. C2 is at global C indices 4..7.
    
    # Part 1: A1 * B1 -> C1
    # U indices 0..3 (A1)
    # V indices 0..7 (B1)
    # W indices 0..7 (C1)
    for i in range(4):
        for c in range(14):
            U[i][c] = U_224[i][c]
            V[i][c] = V_224[i][c]
            W[i][c] = W_224[i][c]
            
    # Part 2: A2 * B2 -> C2
    # U indices 4..7 (A2)
    # V indices 8..15 (B2)
    # W indices 8..15 (C2)
    # Cols 14..27
    for i in range(4):
        for c in range(14):
            U[i + 4][c + 14] = U_224[i][c]
            V[i + 8][c + 14] = V_224[i][c]
            W[i + 8][c + 14] = W_224[i][c]

    return {"U": U, "V": V, "W": W}
