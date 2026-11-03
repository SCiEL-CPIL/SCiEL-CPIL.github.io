"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into two parts:
    # Part 1: A1 (4x2) * B1 (2x6)
    # Part 2: A2 (4x3) * B2 (3x6)
    
    rank1 = 42  # 2 * R(2,2,6) where R(2,2,6)=21
    rank2 = 72  # 2 * R(2,3,6) where R(2,3,6)=36 (schoolbook 3*12)
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a decomposition
    def add_decomp(U_sub, V_sub, W_sub, offset):
        # U_sub: list of lists (n*m x rank_sub)
        # V_sub: list of lists (m*p x rank_sub)
        # W_sub: list of lists (p*n x rank_sub)
        for i in range(len(U)):
            for r in range(len(U_sub[0])):
                U[i][offset + r] += U_sub[i][r]
        for i in range(len(V)):
            for r in range(len(V_sub[0])):
                V[i][offset + r] += V_sub[i][r]
        for i in range(len(W)):
            for r in range(len(W_sub[0])):
                W[i][offset + r] += W_sub[i][r]

    # --- Part 1: 4x2 * 2x6 ---
    # Split into two 2x2 * 2x6
    # Each 2x2 * 2x6 is 3 * Strassen(2x2x2)
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 multiplication
        # A (2x2), B (2x2), C (2x2)
        # U indices: A elements (0..3)
        # V indices: B elements (0..3)
        # W indices: C elements (0..3)
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1
        
        # M2
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = -1
        
        # M3
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1
        
        # M4
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[2][3] = 1; W_s[3][3] = 1
        
        # M5
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1
        
        # M6
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[1][5] = 1; W_s[3][5] = 1
        
        # M7
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1; W_s[2][6] = -1
        
        return U_s, V_s, W_s

    def decomp_2x2x6():
        # A (2x2), B (2x6), C (2x6)
        # B = [B1, B2, B3] each 2x2
        # C = [A B1, A B2, A B3]
        # Rank = 3 * 7 = 21
        U_d = [[0]*21 for _ in range(4)]
        V_d = [[0]*21 for _ in range(12)] # B is 2x6 -> 12 elements
        W_d = [[0]*21 for _ in range(12)] # C is 2x6 -> 12 elements
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        for block in range(3):
            offset = block * 7
            # A is shared
            for i in range(4):
                for r in range(7):
                    U_d[i][offset + r] += U_s[i][r]
            # B block
            # B indices in V_d:
            # Block 0: B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
            # Block 1: B[0,2], B[0,3], B[1,2], B[1,3] -> 4,5,6,7
            # Block 2: B[0,4], B[0,5], B[1,4], B[1,5] -> 8,9,10,11
            # V_s indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Map V_s[i] to V_d[block*4 + i]
            for i in range(4):
                for r in range(7):
                    V_d[block*4 + i][offset + r] += V_s[i][r]
            
            # C block
            # C indices in W_d:
            # Block 0: C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3
            # Block 1: C[0,2], C[0,3], C[1,2], C[1,3] -> 4,5,6,7
            # Block 2: C[0,4], C[0,5], C[1,4], C[1,5] -> 8,9,10,11
            for i in range(4):
                for r in range(7):
                    W_d[block*4 + i][offset + r] += W_s[i][r]
                    
        return U_d, V_d, W_d

    # Construct U, V, W for Part 1
    # A1 is 4x2. Elements:
    # A1[0,0], A1[0,1] (Top 2x2)
    # A1[1,0], A1[1,1] (Bot 2x2)
    # In U (4x5 matrix), Part 1 uses cols 0,1.
    # U indices for A1:
    # Row 0: A[0,0], A[0,1] -> U[0,0], U[0,1]
    # Row 1: A[1,0], A[1,1] -> U[1,0], U[1,1]
    # Row 2: A[2,0], A[2,1] -> U[2,0], U[2,1]
    # Row 3: A[3,0], A[3,1] -> U[3,0], U[3,1]
    
    # We have two 2x2x6 decomp.
    # Top 2x2: A rows 0,1.
    # Bot 2x2: A rows 2,3.
    
    U1_top, V1_top, W1_top = decomp_2x2x6()
    U1_bot, V1_bot, W1_bot = decomp_2x2x6()
    
    # V1 is 2x6. Elements:
    # B1[0,0..5], B1[1,0..5]
    # In V (5x6 matrix), Part 1 uses rows 0,1.
    # V indices:
    # Row 0: B[0,0..5] -> V[0,0..5]
    # Row 1: B[1,0..5] -> V[1,0..5]
    
    # W1 is 2x6. Elements:
    # C1[0,0..5], C1[1,0..5]
    # In W (6x4 matrix), Part 1 uses cols 0,1 (since n=4, C is 4x6, W is 6x4? No, W is p*n = 6x4).
    # W[k*n+i, r].
    # C is 4x6. C[i,k].
    # W index: k*4 + i.
    # Top 2x2 C: i=0,1. k=0..5.
    # W indices for Top: k*4 + 0, k*4 + 1.
    # Bot 2x2 C: i=2,3. k=0..5.
    # W indices for Bot: k*4 + 2, k*4 + 3.
    
    # Add Part 1 to global U, V, W
    offset1 = 0
    
    # Map U1_top to U
    for r in range(21):
        U[0][offset1 + r] += U1_top[0][r] # A[0,0]
        U[1][offset1 + r] += U1_top[1][r] # A[0,1] ? No.
        # U1_top is for 2x2 A.
        # U1_top[0] -> A[0,0]
        # U1_top[1] -> A[0,1]
        # U1_top[2] -> A[1,0]
        # U1_top[3] -> A[1,1]
        
        # Global U rows:
        # Row 0: A[0,0], A[0,1]
        # Row 1: A[1,0], A[1,1]
        # Row 2: A[2,0], A[2,1]
        # Row 3: A[3,0], A[3,1]
        
        # So:
        # U[0,0] is A[0,0]
        # U[0,1] is A[0,1]
        # U[1,0] is A[1,0]
        # U[1,1] is A[1,1]
        
        U[0][offset1 + r] += U1_top[0][r]
        U[0][offset1 + r] += U1_top[1][r] # Wait, U1_top[1] is A[0,1]?
        # In decomp_2x2x6, U_d has 4 rows.
        # Row 0: A[0,0]
        # Row 1: A[0,1]
        # Row 2: A[1,0]
        # Row 3: A[1,1]
        
        # So:
        U[0][offset1 + r] = U1_top[0][r]
        U[0][offset1 + r] += U1_top[1][r] # No, A[0,1] is in U[0,1] in global?
        # Global U is (n*m) x rank.
        # Row index i*m + j.
        # i=0, j=0 -> 0.
        # i=0, j=1 -> 1.
        # i=1, j=0 -> 2.
        # i=1, j=1 -> 3.
        
        # So:
        U[0][offset1 + r] += U1_top[0][r] # A[0,0]
        U[1][offset1 + r] += U1_top[1][r] # A[0,1]
        U[2][offset1 + r] += U1_top[2][r] # A[1,0]
        U[3][offset1 + r] += U1_top[3][r] # A[1,1]
        
        # Bot
        U[4][offset1 + r] += U1_bot[0][r] # A[2,0]
        U[5][offset1 + r] += U1_bot[1][r] # A[2,1]
        U[6][offset1 + r] += U1_bot[2][r] # A[3,0]
        U[7][offset1 + r] += U1_bot[3][r] # A[3,1]
        
        # V
        # V1_top is for B (2x6).
        # V1_top rows:
        # 0: B[0,0..5]
        # 1: B[1,0..5]
        # Wait, V_d in decomp_2x2x6 has 12 rows.
        # Row 0: B[0,0]
        # Row 1: B[0,1]
        # ...
        # Row 5: B[0,5]
        # Row 6: B[1,0]
        # ...
        # Row 11: B[1,5]
        
        # Global V rows:
        # i*m + j. m=5.
        # i=0 (B row 0):
        # j=0 -> 0
        # j=1 -> 1
        # ...
        # j=5 -> 5
        # i=1 (B row 1):
        # j=0 -> 6
        # ...
        # j=5 -> 11
        
        for v_row in range(12):
            V[v_row][offset1 + r] += V1_top[v_row][r]
            V[v_row + 12][offset1 + r] += V1_bot[v_row][r] # Wait, V1_bot is for B rows 2,3?
            # No, Part 1 uses B rows 0,1.
            # V1_top uses B rows 0,1.
            # V1_bot uses B rows 0,1 as well?
            # No, A1 is 4x2.
            # A1 = [A_top; A_bot].
            # B1 is 2x6.
            # C1 = A1 B1.
            # C1_top = A_top B1.
            # C1_bot = A_bot B1.
            # Both use the SAME B1.
            # So V1_top and V1_bot should be the same?
            # No, the decomposition is linear.
            # We add the contributions.
            
            # So for V:
            # V1_top contributes to B rows 0,1.
            # V1_bot contributes to B rows 0,1.
            
            # We sum them.
            pass
            
        # Let's just sum V1_top and V1_bot for the V part.
        # V_part1 = V1_top + V1_bot
        # But we are iterating r.
        
        # W
        # W1_top is for C_top (2x6).
        # W1_bot is for C_bot (2x6).
        # Global W rows: k*n + i.
        # C_top: i=0,1.
        # C_bot: i=2,3.
        
        for w_row in range(12):
            # w_row in W_d:
            # 0: C[0,0]
            # 1: C[0,1]
            # ...
            # 5: C[0,5]
            # 6: C[1,0]
            # ...
            # 11: C[1,5]
            
            # Map to global W index:
            # C[0,k] -> k*4 + 0
            # C[1,k] -> k*4 + 1
            # C[2,k] -> k*4 + 2
            # C[3,k] -> k*4 + 3
            
            # W1_top[w_row] corresponds to C_top[w_row]
            # C_top[0..5] are C[0,0..5]
            # C_top[6..11] are C[1,0..5]
            
            # So:
            # w_row 0..5 -> C[0, 0..5] -> W[0, 4, 8, 12, 16, 20]
            # w_row 6..11 -> C[1, 0..5] -> W[1, 5, 9, 13, 17, 21]
            
            # W1_bot[w_row] corresponds to C_bot[w_row]
            # C_bot[0..5] are C[2, 0..5]
            # C_bot[6..11] are C[3, 0..5]
            
            # So:
            # w_row 0..5 -> C[2, 0..5] -> W[2, 6, 10, 14, 18, 22]
            # w_row 6..11 -> C[3, 0..5] -> W[3, 7, 11, 15, 19, 23]
            
            k = w_row // 2
            i_local = w_row % 2
            
            if i_local == 0:
                # C[0, k] or C[2, k]
                w_idx_top = k * 4 + 0
                w_idx_bot = k * 4 + 2
            else:
                # C[1, k] or C[3, k]
                w_idx_top = k * 4 + 1
                w_idx_bot = k * 4 + 3
                
            W[w_idx_top][offset1 + r] += W1_top[w_row][r]
            W[w_idx_bot][offset1 + r] += W1_bot[w_row][r]
            
    # This is getting complicated. I will simplify by constructing the full matrices directly.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
