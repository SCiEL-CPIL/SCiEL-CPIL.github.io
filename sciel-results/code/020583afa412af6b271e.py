"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create zero matrices
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # 1. Strassen Decomposition for 2x2x2
    def get_strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices for U/V/W are local 0..3
        # A: a(0) b(1) c(2) d(3)
        # B: e(0) f(1) g(2) h(3)
        # C: c00(0) c01(1) c10(2) c11(3)
        
        U = zeros(4, 7)
        V = zeros(4, 7)
        W = zeros(4, 7)
        
        # M1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1 # C11 += M2, C22 -= M2? 
        # Formula: C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        # So W for M2: C21 += 1, C22 += -1.
        # W[2] is C10? No.
        # W index mapping: W[k*n+i].
        # For 2x2x2:
        # k=0, i=0 -> idx 0 (C00)
        # k=0, i=1 -> idx 1 (C01)
        # k=1, i=0 -> idx 2 (C10)
        # k=1, i=1 -> idx 3 (C11)
        
        # Let's re-verify W indices for Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00(1), C11(1) -> W[0]=1, W[3]=1
        # M2: C10(1), C11(-1) -> W[2]=1, W[3]=-1
        # M3: C01(1), C11(1) -> W[1]=1, W[3]=1
        # M4: C00(1), C10(1) -> W[0]=1, W[2]=1
        # M5: C00(-1), C01(1) -> W[0]=-1, W[1]=1
        # M6: C11(1) -> W[3]=1
        # M7: C00(1) -> W[0]=1
        
        # Let's re-fill W correctly.
        W = zeros(4, 7)
        # M1
        W[0][0] = 1; W[3][0] = 1
        # M2
        W[2][1] = 1; W[3][1] = -1
        # M3
        W[1][2] = 1; W[3][2] = 1
        # M4
        W[0][3] = 1; W[2][3] = 1
        # M5
        W[0][4] = -1; W[1][4] = 1
        # M6
        W[3][5] = 1
        # M7
        W[0][6] = 1
        
        # U and V were partially correct, let's verify U/V for M2..M7
        # M2 = (c+d)e -> U[2]=1, U[3]=1, V[0]=1
        # M3 = a(f-h) -> U[0]=1, V[1]=1, V[2]=-1
        # M4 = d(g-e) -> U[3]=1, V[2]=1, V[0]=-1
        # M5 = (a+b)h -> U[0]=1, U[1]=1, V[3]=1
        # M6 = (c-a)(e+f) -> U[2]=1, U[0]=-1, V[0]=1, V[1]=1
        # M7 = (b-d)(g+h) -> U[1]=1, U[3]=-1, V[2]=1, V[3]=1
        
        U = zeros(4, 7)
        V = zeros(4, 7)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[2][2] = -1
        
        # M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        
        return U, V, W

    # 2. Schoolbook Decomposition
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = zeros(n * m, rank)
        V = zeros(m * p, rank)
        W = zeros(p * n, rank)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # 3. Embed a sub-decomposition into the global matrices
    def embed(global_U, global_V, global_W, sub_U, sub_V, sub_W, 
              n_off, m_off, p_off, n_sub, m_sub, p_sub, col_off):
        # sub_U is n_sub * m_sub x R_sub
        # sub_V is m_sub * p_sub x R_sub
        # sub_W is p_sub * n_sub x R_sub
        
        R_sub = len(sub_U[0])
        
        for r in range(R_sub):
            # Embed U
            for i in range(n_sub):
                for j in range(m_sub):
                    global_U[(i + n_off) * m + (j + m_off)][col_off + r] += sub_U[i * m_sub + j][r]
            
            # Embed V
            for j in range(m_sub):
                for k in range(p_sub):
                    global_V[(j + m_off) * p + (k + p_off)][col_off + r] += sub_V[j * p_sub + k][r]
            
            # Embed W
            for k in range(p_sub):
                for i in range(n_sub):
                    global_W[(k + p_off) * n + (i + n_off)][col_off + r] += sub_W[k * n_sub + i][r]

        return col_off + R_sub

    # Initialize Global Matrices
    # We estimate the max rank first to allocate, or just use lists and extend?
    # Lists of lists are easier to extend if we know the total rank.
    # Total Rank = 10 (2x1x5) + 7 (2x2x2) + 7 (2x2x2) + 4 (2x2x1) = 28.
    total_rank = 28
    U = zeros(n * m, total_rank)
    V = zeros(m * p, total_rank)
    W = zeros(p * n, total_rank)
    
    col = 0
    
    # Block 1: 2x1x5 (Schoolbook)
    n1, m1, p1 = 2, 1, 5
    U1, V1, W1 = get_schoolbook(n1, m1, p1)
    col = embed(U, V, W, U1, V1, W1, 0, 0, 0, n1, m1, p1, col)
    
    # Block 2: 2x2x2 (Strassen) - j in 1..2, k in 0..1
    n2, m2, p2 = 2, 2, 2
    U2, V2, W2 = get_strassen_2x2x2()
    col = embed(U, V, W, U2, V2, W2, 0, 1, 0, n2, m2, p2, col)
    
    # Block 3: 2x2x2 (Strassen) - j in 1..2, k in 2..3
    n3, m3, p3 = 2, 2, 2
    U3, V3, W3 = get_strassen_2x2x2()
    col = embed(U, V, W, U3, V3, W3, 0, 1, 2, n3, m3, p3, col)
    
    # Block 4: 2x2x1 (Schoolbook) - j in 1..2, k in 4
    n4, m4, p4 = 2, 2, 1
    U4, V4, W4 = get_schoolbook(n4, m4, p4)
    col = embed(U, V, W, U4, V4, W4, 0, 1, 4, n4, m4, p4, col)
    
    # Verify col matches total_rank
    if col != total_rank:
        # If not, we can truncate or pad, but it should match
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
