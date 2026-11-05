"""Matrix multiplication 4x4 x 4x7 tensor decomposition."""
DIMENSIONS = [4, 4, 7]

import time
import itertools

def _strassen_2x2():
    """Return U,V,W for 2x2 x 2x2 Strassen, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # U rows: 0=a, 1=b, 2=c, 3=d
    # V rows: 0=e, 1=f, 2=g, 3=h
    # W rows: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U col0: a=1,d=1; V col0: e=1,h=1; W: C11+=1, C22+=1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2 = (c+d)e: U col1: c=1,d=1; V col1: e=1; W: C21+=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    # M3 = a(f-h): U col2: a=1; V col2: f=1,h=-1; W: C12+=1, C22-=1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=-1
    # M4 = d(g-e): U col3: d=1; V col3: g=1,e=-1; W: C11-=1, C21+=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=-1; W[2][3]=1
    # M5 = (a+b)h: U col4: a=1,b=1; V col4: h=1; W: C11+=1, C12+=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1; W[1][4]=1
    # M6 = (c-d)(f+g): U col5: c=1,d=-1; V col5: f=1,g=1; W: C22+=1
    U[2][5]=1; U[3][5]=-1; V[1][5]=1; V[2][5]=1; W[3][5]=1
    # M7 = (b-d)(e+g): U col6: b=1,d=-1; V col6: e=1,g=1; W: C11-=1, C22-=1
    U[1][6]=1; U[3][6]=-1; V[0][6]=1; V[2][6]=1; W[0][6]=-1; W[3][6]=-1
    
    return U, V, W

def _schoolbook_2x2x3():
    """Return U,V,W for 2x2 x 2x3 schoolbook, rank 12.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Block decomposition: split rows into 2 groups of 2
    # For each row-group (2x4 submatrix of A) times full B (4x7):
    #   Split B columns into first 4 (2x2x2x2, Strassen) and last 3 (2x2x2x3, schoolbook)
    #   Strassen: rank 7, Schoolbook 2x2x3: rank 12
    #   Per row-group: 7 + 12 = 19
    #   Two row-groups: 38
    
    # But we also need to handle the column structure properly.
    # Actually, let's think of it as: A is 4x4, B is 4x7.
    # Split A by rows: A_top (2x4), A_bot (2x4)
    # Split B by cols: B_left (4x4), B_right (4x3)
    # C_top_left = A_top * B_left (2x2): Strassen, rank 7
    # C_top_right = A_top * B_right (2x3): schoolbook, rank 12
    # C_bot_left = A_bot * B_left (2x2): Strassen, rank 7
    # C_bot_right = A_bot * B_right (2x3): schoolbook, rank 12
    # Total rank: 7+12+7+12 = 38
    
    # U has 4*4=16 rows, V has 4*7=28 rows, W has 7*4=28 rows
    # Wait: U shape (n*m, rank) = (16, rank), V shape (m*p, rank) = (28, rank), W shape (p*n, rank) = (28, rank)
    
    # For each block, we need to map local indices to global.
    
    rank_total = 38
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(28)]
    W = [[0]*rank_total for _ in range(28)]
    
    col_offset = 0
    
    # Block 1: A_top (rows 0-1) * B_left (cols 0-3) -> C top-left (rows 0-1, cols 0-3)
    # Local: A indices 0-3 (i=0,1; j=0,3), B indices 0-3 (j=0,3; k=0,3), C indices 0-3 (k=0,3; i=0,1)
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        for i in range(4):
            if sU[i][c] != 0:
                # Local i -> global row in A: i_local = i, so global A row = i (since top block, rows 0-1)
                # i_local maps to (i_row, i_col) where i_row = i//2, i_col = i%2
                # Global A index: i_row*4 + i_col = (i//2)*4 + i%2
                U[(i//2)*4 + i%2][col_offset + c] = sU[i][c]
        for j in range(4):
            if sV[j][c] != 0:
                # Local j -> global col in B: j_local = j, j_row = j//2, j_col = j%2
                # Global B index: j_row*7 + j_col = (j//2)*7 + j%2
                V[(j//2)*7 + j%2][col_offset + c] = sV[j][c]
        for k in range(4):
            if sW[k][c] != 0:
                # Local k -> global C index: k_local = k, k_row = k//2, k_col = k%2
                # Global C index: k_row*4 + k_col = (k//2)*4 + k%2
                W[(k//2)*4 + k%2][col_offset + c] = sW[k][c]
    col_offset += 7
    
    # Block 2: A_top (rows 0-1) * B_right (cols 4-6) -> C top-right (rows 0-1, cols 4-6)
    sbU, sbV, sbW = _schoolbook_2x2x3()
    for c in range(12):
        for i in range(4):
            if sbU[i][c] != 0:
                U[(i//2)*4 + i%2][col_offset + c] = sbU[i][c]
        for j in range(6):
            if sbV[j][c] != 0:
                # j_local = j, j_row = j//3, j_col = j%3
                # Global B col = 4 + j_col
                V[(j//3)*7 + (4 + j%3)][col_offset + c] = sbV[j][c]
        for k in range(6):
            if sbW[k][c] != 0:
                # k_local = k, k_row = k//2, k_col = k%2
                # Global C: row k_row (0-1), col 4 + k_col
                W[(k//2)*4 + (4 + k%2)][col_offset + c] = sbW[k][c]
    col_offset += 12
    
    # Block 3: A_bot (rows 2-3) * B_left (cols 0-3) -> C bot-left (rows 2-3, cols 0-3)
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        for i in range(4):
            if sU[i][c] != 0:
                # A_bot: global row = 2 + i%2, but i_local = i, i_row = i//2, i_col = i%2
                # Global A index: (2 + i%2) + (i//2)*4? No.
                # A is 4x4. A_bot is rows 2-3. Local i: i_row in {0,1}, i_col in {0,1}
                # Global row = 2 + i_row, global col = i_col
                # Global A index = (2 + i//2)*4 + i%2
                U[(2 + i//2)*4 + i%2][col_offset + c] = sU[i][c]
        for j in range(4):
            if sV[j][c] != 0:
                V[(j//2)*7 + j%2][col_offset + c] = sV[j][c]
        for k in range(4):
            if sW[k][c] != 0:
                # C_bot_left: global row = 2 + k_row, global col = k_col
                # k_local = k, k_row = k//2, k_col = k%2
                W[(2 + k//2)*4 + k%2][col_offset + c] = sW[k][c]
    col_offset += 7
    
    # Block 4: A_bot (rows 2-3) * B_right (cols 4-6) -> C bot-right (rows 2-3, cols 4-6)
    sbU, sbV, sbW = _schoolbook_2x2x3()
    for c in range(12):
        for i in range(4):
            if sbU[i][c] != 0:
                U[(2 + i//2)*4 + i%2][col_offset + c] = sbU[i][c]
        for j in range(6):
            if sbV[j][c] != 0:
                V[(j//3)*7 + (4 + j%3)][col_offset + c] = sbV[j][c]
        for k in range(6):
            if sbW[k][c] != 0:
                W[(2 + k//2)*4 + (4 + k%2)][col_offset + c] = sbW[k][c]
    col_offset += 12
    
    if col_offset != rank_total:
        # Fallback to schoolbook if something went wrong
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
    
    return {"U": U, "V": V, "W": W}
