"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,4,5} into two M_{2,2,5} blocks.
    # A = [A1, A2], B = [B1; B2]
    # C = A1*B1 + A2*B2
    # We use Strassen for the first 2 columns of B (M_{2,2,2}) and Schoolbook for the rest (M_{2,2,3}).
    # This yields rank 7 + 12 = 19 for each M_{2,2,5}.
    # Total rank 38.
    
    def get_strassen_2_2_2():
        # Returns U, V, W for M_{2,2,2} with rank 7
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Mapping (i,j) to index in U: i*2+j
        # Mapping (j,k) to index in V: j*2+k
        # Mapping (k,i) to index in W: k*2+i
        
        # Term 0: M1
        U[0][0] = 1; U[3][0] = 1 # A00, A11
        V[0][0] = 1; V[3][0] = 1 # B00, B11
        W[0][0] = 1; W[3][0] = 1 # C00, C11 (k=0,i=0 and k=1,i=1)
        
        # Term 1: M2
        U[2][1] = 1; U[3][1] = 1 # A10, A11
        V[0][1] = 1 # B00
        W[0][1] = 1 # C00 (k=0,i=0)
        
        # Term 2: M3
        U[0][2] = 1 # A00
        V[2][2] = 1; V[3][2] = -1 # B10, -B11
        W[0][2] = 1 # C00 (k=0,i=0) -- Wait, M3 contributes to C00?
        # M3 = A00(B10-B11).
        # C00 = A00B00 + A01B10. No.
        # C10 = A10B00 + A11B10.
        # C01 = A00B01 + A01B11.
        # C11 = A10B01 + A11B11.
        # Strassen:
        # C00 = M1 - M4 + M7 - M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M7
        # Let's re-verify W indices.
        # W is indexed by (k,i).
        # C00 is i=0, k=0. Index 0.
        # C01 is i=0, k=1. Index 1.
        # C10 is i=1, k=0. Index 2.
        # C11 is i=1, k=1. Index 3.
        
        # Correct W assignments:
        # M1: C00, C11. W[0]=1, W[3]=1.
        # M2: C10. W[2]=1.
        # M3: C01. W[1]=1.
        # M4: C00, C10. W[0]=1, W[2]=1.
        # M5: C01, C11. W[1]=1, W[3]=1.
        # M6: C00. W[0]=1.
        # M7: C00, C11. W[0]=1, W[3]=1.
        
        # Re-do W
        W[0][0] = 1; W[3][0] = 1
        W[2][1] = 1
        W[1][2] = 1
        W[0][3] = 1; W[2][3] = 1
        W[1][4] = 1; W[3][4] = 1
        W[0][5] = 1
        W[0][6] = 1; W[3][6] = 1
        
        # Re-do U, V
        # M1: A00,A11; B00,B11
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        
        # M2: A10,A11; B00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        
        # M3: A00; B10,-B11
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        
        # M4: A11; B01,-B10
        U[3][3]=1
        V[1][3]=1; V[2][3]=-1
        
        # M5: A00,A01; B11
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        
        # M6: A10,-A00; B00,B01
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        
        # M7: A01,-A11; B10,B11
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        
        return U, V, W

    def get_schoolbook_2_2_3():
        # Returns U, V, W for M_{2,2,3} with rank 12
        # i in 0..1, j in 0..1, k in 0..2
        rank = 2*2*3
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W

    # Combine
    rank_total = 7 + 12 + 7 + 12
    U_total = [[0]*rank_total for _ in range(n*m)]
    V_total = [[0]*rank_total for _ in range(m*p)]
    W_total = [[0]*rank_total for _ in range(p*n)]
    
    # Block 1: A1 (cols 0,1 of A), B1 (rows 0,1 of B)
    # A1 is 2x2, B1 is 2x5.
    # We split B1 into B1_1 (2x2) and B1_2 (2x3).
    U1, V1, W1 = get_strassen_2_2_2()
    U2, V2, W2 = get_schoolbook_2_2_3()
    
    # Map indices
    # A1 indices: i*2 + j (j in 0..1) -> global i*4 + j
    # B1 indices: j*5 + k (j in 0..1, k in 0..4)
    #   B1_1: k in 0..1 -> j*5 + k
    #   B1_2: k in 2..4 -> j*5 + k
    # C indices: k*2 + i -> global k*2 + i
    
    offset = 0
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U_total[i*4 + j][offset+c] = U1[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_total[j*5 + k][offset+c] = V1[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_total[k*2 + i][offset+c] = W1[k*2+i][c]
    offset += 7
    
    for c in range(12):
        for i in range(2):
            for j in range(2):
                U_total[i*4 + j][offset+c] = U2[i*2+j][c]
        for j in range(2):
            for k in range(3):
                # k in 2..4 for schoolbook part
                V_total[j*5 + (k+2)][offset+c] = V2[j*3+k][c]
        for k in range(3):
            for i in range(2):
                # k in 2..4 for W
                W_total[(k+2)*2 + i][offset+c] = W2[k*2+i][c]
    offset += 12
    
    # Block 2: A2 (cols 2,3 of A), B2 (rows 2,3 of B)
    U3, V3, W3 = get_strassen_2_2_2()
    U4, V4, W4 = get_schoolbook_2_2_3()
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U_total[i*4 + (j+2)][offset+c] = U3[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_total[(j+2)*5 + k][offset+c] = V3[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_total[k*2 + i][offset+c] = W3[k*2+i][c]
    offset += 7
    
    for c in range(12):
        for i in range(2):
            for j in range(2):
                U_total[i*4 + (j+2)][offset+c] = U4[i*2+j][c]
        for j in range(2):
            for k in range(3):
                V_total[(j+2)*5 + (k+2)][offset+c] = V4[j*3+k][c]
        for k in range(3):
            for i in range(2):
                W_total[(k+2)*2 + i][offset+c] = W4[k*2+i][c]
    offset += 12

    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
