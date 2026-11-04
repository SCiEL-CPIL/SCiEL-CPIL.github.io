import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 78
    # Decomposition:
    # n=3 split into 2+1
    # m=4 split into 2+2
    # p=7 split into 2+2+2+1
    
    # Blocks:
    # 1. A11 (2x2) * B11 (2x7) -> 2x2x7. Rank 25.
    # 2. A12 (2x2) * B21 (2x7) -> 2x2x7. Rank 25.
    # 3. A21 (1x2) * B11 (2x7) -> 1x2x7. Rank 14.
    # 4. A22 (1x2) * B21 (2x7) -> 1x2x7. Rank 14.
    
    total_rank = 25 + 25 + 14 + 14
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        nonlocal col
        for i in range(len(U_block)):
            for k in range(len(U_block[0])):
                if U_block[i][k] != 0:
                    U[u_offset + i][col] += U_block[i][k]
        for i in range(len(V_block)):
            for k in range(len(V_block[0])):
                if V_block[i][k] != 0:
                    V[v_offset + i][col] += V_block[i][k]
        for i in range(len(W_block)):
            for k in range(len(W_block[0])):
                if W_block[i][k] != 0:
                    W[w_offset + i][col] += W_block[i][k]
        col += 1

    # Generate 2x2x2 Strassen block
    def strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        U_b = [[0]*7 for _ in range(4)]
        V_b = [[0]*7 for _ in range(4)]
        W_b = [[0]*7 for _ in range(4)]
        
        # m1 = (a00+a11)(b00+b11)
        U_b[0][0] = 1; U_b[3][0] = 1
        V_b[0][0] = 1; V_b[3][0] = 1
        W_b[0][0] = 1; W_b[1][0] = -1; W_b[2][0] = -1; W_b[3][0] = 1
        
        # m2 = (a10+a11)b00
        U_b[2][1] = 1; U_b[3][1] = 1
        V_b[0][1] = 1
        W_b[2][1] = 1; W_b[3][1] = 1
        
        # m3 = a00(b01-b11)
        U_b[0][2] = 1
        V_b[1][2] = 1; V_b[3][2] = -1
        W_b[0][2] = 1; W_b[1][2] = 1
        
        # m4 = a11(b10-b00)
        U_b[3][3] = 1
        V_b[2][3] = 1; V_b[0][3] = -1
        W_b[2][3] = 1; W_b[3][3] = 1
        
        # m5 = (a00+a01)b11
        U_b[0][4] = 1; U_b[1][4] = 1
        V_b[3][4] = 1
        W_b[1][4] = 1; W_b[3][4] = 1
        
        # m6 = (a10-a00)(b00+b01)
        U_b[2][5] = 1; U_b[0][5] = -1
        V_b[0][5] = 1; V_b[1][5] = 1
        W_b[0][5] = -1; W_b[2][5] = 1
        
        # m7 = (a01-a11)(b10+b11)
        U_b[1][6] = 1; U_b[3][6] = -1
        V_b[2][6] = 1; V_b[3][6] = 1
        W_b[1][6] = -1; W_b[3][6] = 1
        
        return U_b, V_b, W_b

    # Generate 2x2x1 schoolbook block
    def school_221():
        U_b = [[0]*4 for _ in range(4)]
        V_b = [[0]*4 for _ in range(2)]
        W_b = [[0]*4 for _ in range(2)]
        
        # c0 = a00 b0 + a01 b1
        U_b[0][0] = 1; V_b[0][0] = 1; W_b[0][0] = 1
        U_b[1][1] = 1; V_b[1][1] = 1; W_b[0][1] = 1
        
        # c1 = a10 b0 + a11 b1
        U_b[2][2] = 1; V_b[0][2] = 1; W_b[1][2] = 1
        U_b[3][3] = 1; V_b[1][3] = 1; W_b[1][3] = 1
        
        return U_b, V_b, W_b

    # Generate 1x2x2 schoolbook block
    def school_122():
        U_b = [[0]*4 for _ in range(2)]
        V_b = [[0]*4 for _ in range(4)]
        W_b = [[0]*4 for _ in range(2)]
        
        # c0 = a0 b00 + a1 b10
        U_b[0][0] = 1; V_b[0][0] = 1; W_b[0][0] = 1
        U_b[1][1] = 1; V_b[2][1] = 1; W_b[0][1] = 1
        
        # c1 = a0 b01 + a1 b11
        U_b[0][2] = 1; V_b[1][2] = 1; W_b[1][2] = 1
        U_b[1][3] = 1; V_b[3][3] = 1; W_b[1][3] = 1
        
        return U_b, V_b, W_b

    # Generate 1x2x1 schoolbook block
    def school_121():
        U_b = [[0]*2 for _ in range(2)]
        V_b = [[0]*2 for _ in range(2)]
        W_b = [[0]*2 for _ in range(1)]
        
        U_b[0][0] = 1; V_b[0][0] = 1; W_b[0][0] = 1
        U_b[1][1] = 1; V_b[1][1] = 1; W_b[0][1] = 1
        
        return U_b, V_b, W_b

    # Block 1: A11 (2x2) * B11 (2x7)
    # A11 is rows 0-1, cols 0-1 of A. Offset U: 0
    # B11 is rows 0-1, cols 0-6 of B. Offset V: 0
    # C11 is rows 0-1, cols 0-6 of C. Offset W: 0
    # Split p=7 into 2,2,2,1
    # p_offsets: 0, 2, 4, 6
    
    # 1a: 2x2x2 (p 0-1)
    Ub, Vb, Wb = strassen_222()
    add_block(Ub, Vb, Wb, 0, 0, 0)
    
    # 1b: 2x2x2 (p 2-3)
    Ub, Vb, Wb = strassen_222()
    add_block(Ub, Vb, Wb, 0, 2, 2)
    
    # 1c: 2x2x2 (p 4-5)
    Ub, Vb, Wb = strassen_222()
    add_block(Ub, Vb, Wb, 0, 4, 4)
    
    # 1d: 2x2x1 (p 6)
    Ub, Vb, Wb = school_221()
    add_block(Ub, Vb, Wb, 0, 6, 6)

    # Block 2: A12 (2x2) * B21 (2x7)
    # A12 is rows 0-1, cols 2-3 of A. Offset U: 2
    # B21 is rows 2-3, cols 0-6 of B. Offset V: 14 (2*7)
    # C12 is rows 0-1, cols 0-6 of C. Offset W: 0 (Same C rows)
    # Wait, C is p*n. C[0:2, :] corresponds to W[0:2*3]?
    # W index: k*n + i.
    # C[i,k] is at W[k*n+i].
    # For i in 0..1, k in 0..6.
    # W offset for i=0,1 is 0.
    
    # 2a: 2x2x2 (p 0-1)
    Ub, Vb, Wb = strassen_222()
    add_block(Ub, Vb, Wb, 2, 14, 0)
    
    # 2b: 2x2x2 (p 2-3)
    Ub, Vb, Wb = strassen_222()
    add_block(Ub, Vb, Wb, 2, 16, 2)
    
    # 2c: 2x2x2 (p 4-5)
    Ub, Vb, Wb = strassen_222()
    add_block(Ub, Vb, Wb, 2, 18, 4)
    
    # 2d: 2x2x1 (p 6)
    Ub, Vb, Wb = school_221()
    add_block(Ub, Vb, Wb, 2, 20, 6)

    # Block 3: A21 (1x2) * B11 (2x7)
    # A21 is row 2, cols 0-1 of A. Offset U: 8 (2*4)
    # B11 is rows 0-1, cols 0-6 of B. Offset V: 0
    # C21 is row 2, cols 0-6 of C. Offset W: 18 (6*3)
    
    # 3a: 1x2x2 (p 0-1)
    Ub, Vb, Wb = school_122()
    add_block(Ub, Vb, Wb, 8, 0, 18)
    
    # 3b: 1x2x2 (p 2-3)
    Ub, Vb, Wb = school_122()
    add_block(Ub, Vb, Wb, 8, 2, 20)
    
    # 3c: 1x2x2 (p 4-5)
    Ub, Vb, Wb = school_122()
    add_block(Ub, Vb, Wb, 8, 4, 22)
    
    # 3d: 1x2x1 (p 6)
    Ub, Vb, Wb = school_121()
    add_block(Ub, Vb, Wb, 8, 6, 24)

    # Block 4: A22 (1x2) * B21 (2x7)
    # A22 is row 2, cols 2-3 of A. Offset U: 10
    # B21 is rows 2-3, cols 0-6 of B. Offset V: 14
    # C22 is row 2, cols 0-6 of C. Offset W: 18
    
    # 4a: 1x2x2 (p 0-1)
    Ub, Vb, Wb = school_122()
    add_block(Ub, Vb, Wb, 10, 14, 18)
    
    # 4b: 1x2x2 (p 2-3)
    Ub, Vb, Wb = school_122()
    add_block(Ub, Vb, Wb, 10, 16, 20)
    
    # 4c: 1x2x2 (p 4-5)
    Ub, Vb, Wb = school_122()
    add_block(Ub, Vb, Wb, 10, 18, 22)
    
    # 4d: 1x2x1 (p 6)
    Ub, Vb, Wb = school_121()
    add_block(Ub, Vb, Wb, 10, 20, 24)

    return {"U": U, "V": V, "W": W}

