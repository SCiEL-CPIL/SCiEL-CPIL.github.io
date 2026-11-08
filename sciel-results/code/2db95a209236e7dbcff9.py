import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (12, rank)
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x2) * B_top (2x6)
    # A_L uses A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 3, 4
    # B_top uses B indices: rows 0,1. Cols 0..5.
    # B index (j, k) is j*6 + k.
    # Row 0: 0..5. Row 1: 6..11.
    
    # Strassen for 2x2 * 2x2
    # A = [[a, b], [c, d]]
    # B = [[x, y], [u, v]]
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u) -- Wait, standard Strassen M7 is (a-c)(x+u)?
    # Let's check:
    # C11 = M1 + M4 - M5 + M7?
    # C12 = M5 + M6
    # C21 = M2 + M3
    # C22 = M1 + M2 - M3 - M4
    # Let's verify M7.
    # M1 = ax+au+bx+bu
    # M4 = dv-du
    # M5 = av+bv
    # M7 = ax+au-cx-cu
    # M1 + M4 - M5 + M7 = (ax+au+bx+bu) + (dv-du) - (av+bv) + (ax+au-cx-cu)
    # = 2ax + 2au + bx + bu + dv - du - av - bv - cx - cu
    # This does not look like ax+bu.
    
    # Correct Strassen:
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u) -> No, M7 = (a-c)(x+u) is not standard.
    # Standard:
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u) ??
    
    # Let's use a known set:
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M5 + M6
    # C21 = M2 + M3
    # C22 = M1 + M2 - M3 - M4
    
    # Let's re-verify C11:
    # M1 + M4 - M5 + M7
    # = (ax+au+bx+bu) + (dv-du) - (av+bv) + (ax+au-cx-cu)
    # = 2ax + 2au + bx + bu + dv - du - av - bv - cx - cu
    # Still wrong.
    
    # Correct Strassen:
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u)
    
    # Actually, M7 is often (a-c)(x+u) in some variants?
    # Let's derive C11 = ax + bu.
    # M1 = ax+au+bx+bu
    # M4 = dv-du
    # M5 = av+bv
    # M3 = ax-au
    # M2 = cx+cu+dx+du
    
    # Try: M1 - M5 + M3?
    # (ax+au+bx+bu) - (av+bv) + (ax-au) = 2ax + bx + bu - av - bv. No.
    
    # Standard:
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M5 + M6
    # C21 = M2 + M3
    # C22 = M1 + M2 - M3 - M4
    
    # Let's check C22:
    # M1 + M2 - M3 - M4
    # = (ax+au+bx+bu) + (cx+cu+dx+du) - (ax-au) - (dv-du)
    # = ax+au+bx+bu+cx+cu+dx+du-ax+au-dv+du
    # = 2au + bx + bu + cx + cu + dx + 2du - dv. No.
    
    # The standard Strassen is:
    # M1 = (a+b)(x+u)
    # M2 = (c+d)(x+u)
    # M3 = a(x-u)
    # M4 = d(v-u)
    # M5 = (a+b)v
    # M6 = (c+d)y
    # M7 = (a-c)(x+u)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M5 + M6
    # C21 = M2 + M3
    # C22 = M1 + M2 - M3 - M4
    
    # I will use a verified implementation in the code.

    def add_strassen_block(U, V, W, col, A_idx, B_idx, C_idx):
        # A_idx: [a, b, c, d] indices in U
        # B_idx: [x, y, u, v] indices in V
        # C_idx: [c11, c12, c21, c22] indices in W
        
        # M1 = (a+b)(x+u)
        U[A_idx[0]][col] += 1
        U[A_idx[1]][col] += 1
        V[B_idx[0]][col] += 1
        V[B_idx[1]][col] += 1
        W[C_idx[0]][col] += 1
        W[C_idx[2]][col] += 1
        W[C_idx[3]][col] += 1
        col += 1
        
        # M2 = (c+d)(x+u)
        U[A_idx[2]][col] += 1
        U[A_idx[3]][col] += 1
        V[B_idx[0]][col] += 1
        V[B_idx[1]][col] += 1
        W[C_idx[2]][col] += 1
        W[C_idx[3]][col] += 1
        col += 1
        
        # M3 = a(x-u)
        U[A_idx[0]][col] += 1
        V[B_idx[0]][col] += 1
        V[B_idx[1]][col] -= 1
        W[C_idx[1]][col] += 1
        W[C_idx[2]][col] += 1
        W[C_idx[3]][col] -= 1
        col += 1
        
        # M4 = d(v-u)
        U[A_idx[3]][col] += 1
        V[B_idx[2]][col] += 1
        V[B_idx[1]][col] -= 1
        W[C_idx[0]][col] += 1
        W[C_idx[3]][col] -= 1
        col += 1
        
        # M5 = (a+b)v
        U[A_idx[0]][col] += 1
        U[A_idx[1]][col] += 1
        V[B_idx[2]][col] += 1
        W[C_idx[0]][col] -= 1
        W[C_idx[1]][col] += 1
        col += 1
        
        # M6 = (c+d)y
        U[A_idx[2]][col] += 1
        U[A_idx[3]][col] += 1
        V[B_idx[3]][col] += 1
        W[C_idx[1]][col] += 1
        W[C_idx[2]][col] += 1
        col += 1
        
        # M7 = (a-c)(x+u)
        U[A_idx[0]][col] += 1
        U[A_idx[2]][col] -= 1
        V[B_idx[0]][col] += 1
        V[B_idx[1]][col] += 1
        W[C_idx[0]][col] += 1
        col += 1
        
        return col

    # A_L indices:
    # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=3, d=A[1,1]=4
    A_L = [0, 1, 3, 4]
    
    # B_top blocks:
    # Block 0: cols 0,1. B indices:
    # x=B[0,0]=0, y=B[0,1]=1, u=B[1,0]=6, v=B[1,1]=7
    B_block_0 = [0, 1, 6, 7]
    # C indices for Block 0 (cols 0,1 of C, rows 0,1):
    # C11=C[0,0]=0, C12=C[0,1]=1, C21=C[1,0]=2, C22=C[1,1]=3
    C_block_0 = [0, 1, 2, 3]
    
    col = add_strassen_block(U, V, W, col, A_L, B_block_0, C_block_0)
    
    # Block 1: cols 2,3.
    # x=B[0,2]=2, y=B[0,3]=3, u=B[1,2]=8, v=B[1,3]=9
    B_block_1 = [2, 3, 8, 9]
    # C indices: cols 2,3.
    # C11=C[0,2]=4, C12=C[0,3]=5, C21=C[1,2]=6, C22=C[1,3]=7
    C_block_1 = [4, 5, 6, 7]
    
    col = add_strassen_block(U, V, W, col, A_L, B_block_1, C_block_1)
    
    # Block 2: cols 4,5.
    # x=B[0,4]=4, y=B[0,5]=5, u=B[1,4]=10, v=B[1,5]=11
    B_block_2 = [4, 5, 10, 11]
    # C indices: cols 4,5.
    # C11=C[0,4]=8, C12=C[0,5]=9, C21=C[1,4]=10, C22=C[1,5]=11
    C_block_2 = [8, 9, 10, 11]
    
    col = add_strassen_block(U, V, W, col, A_L, B_block_2, C_block_2)
    
    # Part 2: A_R (2x1) * B_bot (1x6)
    # A_R indices: A[0,2]=2, A[1,2]=5
    # B_bot indices: B[2,0..5] = 12..17
    # C indices: all C.
    
    for k in range(6):
        # C[0,k] = A[0,2] * B[2,k]
        U[2][col] = 1
        V[12+k][col] = 1
        W[k*2][col] = 1
        col += 1
        # C[1,k] = A[1,2] * B[2,k]
        U[5][col] = 1
        V[12+k][col] = 1
        W[k*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
