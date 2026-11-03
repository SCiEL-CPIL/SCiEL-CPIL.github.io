"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get 2x2x2 Strassen decomposition
    def strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i,j], V[j,k], W[k,i]
        # 0: a, 1: b, 2: c, 3: d
        # 0: e, 1: f, 2: g, 3: h
        # 0: c11, 1: c12, 2: c21, 3: c22
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1 # C11+=M1, C22-=M1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's fill W based on the C formulas.
        # W[k,i] corresponds to C[i,k]?
        # Problem says W[k*n+i, r]. For 2x2, n=2.
        # W[0*2+0]=W00 (C00), W[1*2+0]=W10 (C10), W[0*2+1]=W01 (C01), W[1*2+1]=W11 (C11).
        # My indices above were C11, C12, C21, C22.
        # C11 is (0,0). C12 is (0,1). C21 is (1,0). C22 is (1,1).
        
        # M1 contributes to C11 (+1) and C22 (+1)
        W[0][0] += 1 # C00
        W[3][0] += 1 # C11
        
        # M2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] += 1 # C10 (C21)
        W[3][1] += -1 # C11 (C22)
        
        # M3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] += 1 # C01 (C12)
        W[3][2] += 1 # C11 (C22)
        
        # M4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] += 1 # C00 (C11)
        W[2][3] += 1 # C10 (C21)
        
        # M5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] += -1 # C00 (C11)
        W[1][4] += 1 # C01 (C12)
        
        # M6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] += 1 # C11 (C22)
        
        # M7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] += 1 # C00 (C11)
        
        return U, V, W

    # Helper to build 4x4x4 from 2x2x2
    def build_4x4x4():
        # A = [[A1, A2], [A3, A4]]
        # B = [[B1, B2], [B3, B4]]
        # C = [[C1, C2], [C3, C4]]
        # C1 = A1B1 + A2B3
        # C2 = A1B2 + A2B4
        # C3 = A3B1 + A4B3
        # C4 = A3B2 + A4B4
        
        # We use Strassen on the 2x2 block structure.
        # The 7 products are:
        # M1 = (A1+A4)(B1+B4)
        # M2 = (A3+A4)B1
        # M3 = A1(B2-B4)
        # M4 = A4(B3-B1)
        # M5 = (A1+A2)B4
        # M6 = (A3-A1)(B1+B2)
        # M7 = (A2-A4)(B3+B4)
        
        # C1 = M1 + M4 - M5 + M7
        # C2 = M3 + M5
        # C3 = M2 + M4
        # C4 = M1 - M2 + M3 + M6
        
        # Each Mk is a 2x2x2 multiplication.
        # We have the 2x2x2 decomposition U2, V2, W2.
        U2, V2, W2 = strassen_2x2x2()
        
        # Now we need to map the 4x4 indices to the 2x2 block indices.
        # A indices: 0,1,2,3 -> 00,01,10,11
        # 0: A1_00, 1: A1_01, 2: A1_10, 3: A1_11? No.
        # A is 4x4.
        # A1 is top-left 2x2. Indices 0,1,4,5?
        # Row 0: 0,1. Row 1: 2,3.
        # A1: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5.
        # A2: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7.
        # A3: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13.
        # A4: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15.
        
        # This mapping is getting messy.
        # I will just return the schoolbook for now to ensure correctness, 
        # and then try to improve.
        
        # Actually, I can just use the schoolbook for 4x4x4 and 4x4x8.
        # Rank 128.
        # But I want to improve.
        
        # Let's just return the schoolbook.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W

    U, V, W = build_4x4x4()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
