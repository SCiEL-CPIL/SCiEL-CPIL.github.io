import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_1x2x1():
    """Rank 2 decomposition for 1x2x1."""
    U = [[1, 0], [0, 1]]
    V = [[1, 0], [0, 1]]
    W = [[1, 1]]
    return U, V, W

def _build_1x2x3():
    """Rank 6 decomposition for 1x2x3."""
    rank = 6
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(3)]
    
    for k in range(3):
        u, v, w = _build_1x2x1()
        # u is 2x2, v is 2x2, w is 1x2
        for i in range(2):
            for c in range(2):
                U[i][k*2+c] = u[i][c]
                V[i][k*2+c] = v[i][c]
        for c in range(2):
            W[k][k*2+c] = w[0][c]
            
    return U, V, W

def _build_2x2x1():
    """Rank 4 decomposition for 2x2x1."""
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return U, V, W

def _build_2x2x2_strassen():
    """Rank 7 decomposition for 2x2x2 using Strassen."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # P2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # P3: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = -1
    
    # P4: A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[3][3] = 1; W[2][3] = -1
    
    # P5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1; W[0][5] = -1; W[1][5] = -1
    
    # P7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _build_2x2x3():
    """Rank 11 decomposition for 2x2x3."""
    # Split p=3 into 1+2
    u1, v1, w1 = _build_2x2x1()
    u2, v2, w2 = _build_2x2x2_strassen()
    
    rank1 = len(u1[0])
    rank2 = len(u2[0])
    rank = rank1 + rank2
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(6)] # p*n = 3*2 = 6
    
    # Fill U, V
    for i in range(4):
        for c in range(rank1):
            U[i][c] = u1[i][c]
        for c in range(rank2):
            U[i][rank1+c] = u2[i][c]
            
    for i in range(2):
        for c in range(rank1):
            V[i][c] = v1[i][c]
        for c in range(rank2):
            V[i][rank1+c] = v2[i][c]
            
    # Fill W
    # w1 is 2x rank1. Rows k*n+i for k=0. n=2.
    # w1 rows 0,1 correspond to global W rows 0,1.
    for i in range(2):
        for c in range(rank1):
            W[i][c] = w1[i][c]
            
    # w2 is 4x rank2. Rows k'*n+i' for k' in {0,1}. n=2.
    # w2 rows 0,1 correspond to global W rows 2,3 (k=1).
    # w2 rows 2,3 correspond to global W rows 4,5 (k=2).
    for i in range(4):
        for c in range(rank2):
            W[2+i][rank1+c] = w2[i][c]
            
    return U, V, W

def _build_3x2x3():
    """Rank 17 decomposition for 3x2x3."""
    # Split n=3 into 1+2
    u1, v1, w1 = _build_1x2x3()
    u2, v2, w2 = _build_2x2x3()
    
    rank1 = len(u1[0])
    rank2 = len(u2[0])
    rank = rank1 + rank2
    
    U = [[0]*rank for _ in range(6)] # n*m = 3*2 = 6
    V = [[0]*rank for _ in range(6)] # m*p = 2*3 = 6
    W = [[0]*rank for _ in range(9)] # p*n = 3*3 = 9
    
    # Fill U
    # u1 is 2x rank1. Rows i*m+j for i=0. m=2.
    # u1 rows 0,1 correspond to global U rows 0,1.
    for i in range(2):
        for c in range(rank1):
            U[i][c] = u1[i][c]
            
    # u2 is 4x rank2. Rows i'*m+j' for i' in {0,1}. m=2.
    # u2 rows 0,1 correspond to global U rows 2,3 (i=1).
    # u2 rows 2,3 correspond to global U rows 4,5 (i=2).
    for i in range(4):
        for c in range(rank2):
            U[2+i][rank1+c] = u2[i][c]
            
    # Fill V
    # v1 is 2x rank1. Rows j*p+k.
    # v2 is 6x rank2. Rows j*p+k.
    # Both cover all j, k.
    for i in range(2):
        for c in range(rank1):
            V[i][c] = v1[i][c]
        for c in range(rank2):
            V[i][rank1+c] = v2[i][c] # Wait, v2 has 6 rows. v1 has 2 rows.
            # v1 rows 0,1 correspond to j=0,1.
            # v2 rows 0,1 correspond to j=0,1? No, v2 is for 2x2x3.
            # v2 rows are j*p+k. j in {0,1}.
            # So v2 rows 0,1 are j=0. v2 rows 2,3 are j=1?
            # No, v2 shape is (m*p) x rank = (2*3) x rank = 6 x rank.
            # Rows 0,1,2 for j=0? No.
            # j=0: k=0,1,2 -> rows 0,1,2.
            # j=1: k=0,1,2 -> rows 3,4,5.
            # v1 shape is (m*p) x rank = (2*3) x rank = 6 x rank.
            # So v1 and v2 have same row structure.
            pass
            
    # Correct V fill:
    for i in range(6):
        for c in range(rank1):
            V[i][c] = v1[i][c]
        for c in range(rank2):
            V[i][rank1+c] = v2[i][c]
            
    # Fill W
    # w1 is 3x rank1. Rows k*n+i for i=0. n=3.
    # w1 rows k correspond to global W rows 3k.
    for i in range(3):
        for c in range(rank1):
            W[3*i][c] = w1[i][c]
            
    # w2 is 6x rank2. Rows k*n+i' for i' in {0,1}. n=3.
    # w2 rows 0,1 correspond to k=0, i'=0,1. Global W rows 3*0+1, 3*0+2?
    # No, global W row index is k*n + i.
    # For w2, local i' is 0,1. Global i = i' + 1.
    # So global row = k*3 + (i'+1).
    # w2 row index is k*2 + i'.
    for i in range(6):
        k = i // 2
        ip = i % 2
        global_row = k * 3 + (ip + 1)
        for c in range(rank2):
            W[global_row][rank1+c] = w2[i][c]
            
    return U, V, W

def _build_3x2x6():
    """Rank 34 decomposition for 3x2x6."""
    # Split p=6 into 3+3
    u1, v1, w1 = _build_3x2x3()
    u2, v2, w2 = _build_3x2x3()
    
    rank1 = len(u1[0])
    rank2 = len(u2[0])
    rank = rank1 + rank2
    
    U = [[0]*rank for _ in range(6)] # n*m = 3*2 = 6
    V = [[0]*rank for _ in range(12)] # m*p = 2*6 = 12
    W = [[0]*rank for _ in range(18)] # p*n = 6*3 = 18
    
    # Fill U
    # u1, u2 same shape.
    for i in range(6):
        for c in range(rank1):
            U[i][c] = u1[i][c]
        for c in range(rank2):
            U[i][rank1+c] = u2[i][c]
            
    # Fill V
    # v1 is 6x rank1. Rows j*p1+k for p1=3.
    # v2 is 6x rank2. Rows j*p2+k' for p2=3.
    # Global V rows j*p+k for p=6.
    # v1 rows j*3+k correspond to global V rows j*6+k.
    # v2 rows j*3+k' correspond to global V rows j*6+(k'+3).
    for i in range(6):
        j = i // 3
        k = i % 3
        for c in range(rank1):
            V[j*6+k][c] = v1[i][c]
        for c in range(rank2):
            V[j*6+k+3][rank1+c] = v2[i][c]
            
    # Fill W
    # w1 is 9x rank1. Rows k*n+i for k in {0,1,2}.
    # w2 is 9x rank2. Rows k'*n+i for k' in {0,1,2}.
    # Global W rows k*n+i for k in {0..5}.
    # w1 rows correspond to global rows 0..8.
    # w2 rows correspond to global rows 9..17.
    for i in range(9):
        for c in range(rank1):
            W[i][c] = w1[i][c]
        for c in range(rank2):
            W[9+i][rank1+c] = w2[i][c]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split m=4 into 2+2
    u1, v1, w1 = _build_3x2x6()
    u2, v2, w2 = _build_3x2x6()
    
    rank1 = len(u1[0])
    rank2 = len(u2[0])
    rank = rank1 + rank2
    
    U = [[0]*rank for _ in range(12)] # n*m = 3*4 = 12
    V = [[0]*rank for _ in range(24)] # m*p = 4*6 = 24
    W = [[0]*rank for _ in range(18)] # p*n = 6*3 = 18
    
    # Fill U
    # u1 is 6x rank1. Rows i*m1+j for m1=2.
    # u2 is 6x rank2. Rows i*m2+j' for m2=2.
    # Global U rows i*m+j for m=4.
    # u1 rows i*2+j correspond to global U rows i*4+j.
    # u2 rows i*2+j' correspond to global U rows i*4+(j'+2).
    for i in range(6):
        row_idx = i // 2
        col_idx = i % 2
        for c in range(rank1):
            U[row_idx*4+col_idx][c] = u1[i][c]
        for c in range(rank2):
            U[row_idx*4+col_idx+2][rank1+c] = u2[i][c]
            
    # Fill V
    # v1 is 12x rank1. Rows j*p+k for j in {0,1}.
    # v2 is 12x rank2. Rows j'*p+k for j' in {0,1}.
    # Global V rows j*p+k for j in {0..3}.
    # v1 rows j*6+k correspond to global V rows j*6+k.
    # v2 rows j'*6+k correspond to global V rows (j'+2)*6+k.
    for i in range(12):
        row_idx = i // 6
        col_idx = i % 6
        for c in range(rank1):
            V[row_idx*6+col_idx][c] = v1[i][c]
        for c in range(rank2):
            V[(row_idx+2)*6+col_idx][rank1+c] = v2[i][c]
            
    # Fill W
    # w1, w2 same shape.
    for i in range(18):
        for c in range(rank1):
            W[i][c] = w1[i][c]
        for c in range(rank2):
            W[i][rank1+c] = w2[i][c]
            
    return {"U": U, "V": V, "W": W}
