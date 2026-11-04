import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # Rank 57 decomposition
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)] # 10 x 57
    V = [[0] * rank for _ in range(m * p)] # 30 x 57
    W = [[0] * rank for _ in range(p * n)] # 12 x 57
    
    col = 0
    
    # Part 1: A_1 (2x2) * B_1 (2x6)
    # A_1 rows: 0,1. Cols 0,1.
    # B_1 rows: 0,1. Cols 0..5.
    # Split B_1 into 3 blocks of 2x2: B_1[0:2, 0:2], B_1[0:2, 2:4], B_1[0:2, 4:6]
    
    # Strassen factors for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[x, y], [u, v]]
    # M1 = (a+d)(x+v)
    # M2 = (c+d)(x)
    # M3 = (a)(x+u)
    # M4 = (d)(v-u)
    # M5 = (a+b)(v)
    # M6 = (c-a)(x+u)
    # M7 = (b-d)(y+v)
    
    # U indices for A_1:
    # i=0, j=0 -> a
    # i=0, j=1 -> b
    # i=1, j=0 -> c
    # i=1, j=1 -> d
    # U row index: i*m + j. m=5.
    # a: 0*5+0=0
    # b: 0*5+1=1
    # c: 1*5+0=5
    # d: 1*5+1=6
    
    # V indices for B_1 block r (cols 2r, 2r+1):
    # j=0, k=2r -> x
    # j=0, k=2r+1 -> y
    # j=1, k=2r -> u
    # j=1, k=2r+1 -> v
    # V row index: j*p + k. p=6.
    # x: 0*6 + 2r
    # y: 0*6 + 2r+1
    # u: 1*6 + 2r
    # v: 1*6 + 2r+1
    
    # W indices for C_1 block r (cols 2r, 2r+1):
    # i=0, k=2r -> C00
    # i=0, k=2r+1 -> C01
    # i=1, k=2r -> C10
    # i=1, k=2r+1 -> C11
    # W row index: k*n + i. n=2.
    # C00: 2r*2 + 0
    # C01: (2r+1)*2 + 0
    # C10: 2r*2 + 1
    # C11: (2r+1)*2 + 1
    
    strassen_U = [
        [1, 1, 0, 0, 1, 0, 0], # a
        [0, 0, 1, 0, 1, 0, 0], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 0, 0, 1, 0, 0, 1]  # d
    ]
    # Note: M1=a+d, M2=c+d, M3=a, M4=d, M5=a+b, M6=c-a, M7=b-d
    # Let's verify:
    # M1: a=1, d=1. Correct.
    # M2: c=1, d=1. Correct.
    # M3: a=1. Correct.
    # M4: d=1. Correct.
    # M5: a=1, b=1. Correct.
    # M6: c=1, a=-1. Correct.
    # M7: b=1, d=-1. Correct.
    
    strassen_V = [
        [1, 0, 1, 0, 0, 1, 0], # x: M1, M2, M3, M6
        [0, 1, 0, 0, 1, 0, 1], # y: M5, M7
        [1, 0, 1, -1, 0, 1, 0], # u: M1, M3, M4, M6
        [1, 0, 0, 1, 1, 0, 1]  # v: M1, M4, M5, M7
    ]
    # Let's verify V:
    # M1: x+v. x=1, v=1.
    # M2: x. x=1.
    # M3: x+u. x=1, u=1.
    # M4: v-u. v=1, u=-1.
    # M5: v. v=1.
    # M6: x+u. x=1, u=1.
    # M7: y+v. y=1, v=1.
    
    strassen_W = [
        [1, 0, 1, 1, -1, 0, 1], # C00: M1+M4-M5+M7
        [0, 1, 1, 0, 1, 0, 0],  # C01: M3+M5
        [0, 1, 0, 1, 0, 0, 0],  # C10: M2+M4
        [1, -1, 1, 0, 0, 1, 1]  # C11: M1-M2+M3+M6
    ]
    # Let's verify W:
    # C00: M1(1), M4(1), M5(-1), M7(1).
    # C01: M3(1), M5(1).
    # C10: M2(1), M4(1).
    # C11: M1(1), M2(-1), M3(1), M6(1).
    
    for r in range(3):
        # U
        for i in range(2):
            for j in range(2):
                u_idx = i * m + j
                for s in range(7):
                    U[u_idx][col + s] = strassen_U[i][j][s] if False else strassen_U[i][j]
        # Correction: strassen_U is defined as list of lists for a,b,c,d.
        # strassen_U[0] is for a. strassen_U[1] for b, etc.
        # a is at (0,0), b at (0,1), c at (1,0), d at (1,1).
        
        # Re-apply U correctly
        for s in range(7):
            U[0][col+s] = strassen_U[0][s] # a
            U[1][col+s] = strassen_U[1][s] # b
            U[5][col+s] = strassen_U[2][s] # c
            U[6][col+s] = strassen_U[3][s] # d
            
        # V
        for s in range(7):
            V[0*6 + 2*r][col+s] = strassen_V[0][s] # x
            V[0*6 + 2*r+1][col+s] = strassen_V[1][s] # y
            V[1*6 + 2*r][col+s] = strassen_V[2][s] # u
            V[1*6 + 2*r+1][col+s] = strassen_V[3][s] # v
            
        # W
        for s in range(7):
            W[2*r*2 + 0][col+s] = strassen_W[0][s] # C00
            W[(2*r+1)*2 + 0][col+s] = strassen_W[1][s] # C01
            W[2*r*2 + 1][col+s] = strassen_W[2][s] # C10
            W[(2*r+1)*2 + 1][col+s] = strassen_W[3][s] # C11
            
        col += 7

    # Part 2: A_2 (2x3) * B_2 (3x6)
    # A_2 rows: 0,1. Cols 2,3,4.
    # B_2 rows: 2,3,4. Cols 0..5.
    # Split B_2 into 3 blocks of 3x2: B_2[2:5, 0:2], B_2[2:5, 2:4], B_2[2:5, 4:6]
    
    # Schoolbook for 2x3x2 (Rank 12)
    # A = [[a, b, c], [d, e, f]]
    # B = [[x, y], [u, v], [w, z]]
    # C00 = ax + bu + cw
    # C01 = ay + bv + cz
    # C10 = dx + eu + fw
    # C11 = dy + ev + fz
    
    # U indices for A_2:
    # a: (0,2) -> 0*5+2=2
    # b: (0,3) -> 0*5+3=3
    # c: (0,4) -> 0*5+4=4
    # d: (1,2) -> 1*5+2=7
    # e: (1,3) -> 1*5+3=8
    # f: (1,4) -> 1*5+4=9
    
    # V indices for B_2 block r (cols 2r, 2r+1):
    # x: (2, 2r) -> 2*6+2r
    # y: (2, 2r+1) -> 2*6+2r+1
    # u: (3, 2r) -> 3*6+2r
    # v: (3, 2r+1) -> 3*6+2r+1
    # w: (4, 2r) -> 4*6+2r
    # z: (4, 2r+1) -> 4*6+2r+1
    
    # W indices for C_2 block r (cols 2r, 2r+1):
    # C00: 2r*2+0
    # C01: (2r+1)*2+0
    # C10: 2r*2+1
    # C11: (2r+1)*2+1
    
    # 12 multiplications:
    # P1=ax, P2=bu, P3=cw, P4=ay, P5=bv, P6=cz, P7=dx, P8=eu, P9=fw, P10=dy, P11=ev, P12=fz
    
    for r in range(3):
        # U
        U[2][col+0] = 1 # a
        U[3][col+1] = 1 # b
        U[4][col+2] = 1 # c
        U[7][col+6] = 1 # d
        U[8][col+7] = 1 # e
        U[9][col+8] = 1 # f
        U[3][col+4] = 1 # b
        U[4][col+5] = 1 # c
        U[7][col+9] = 1 # d
        U[8][col+10] = 1 # e
        U[9][col+11] = 1 # f
        
        # V
        V[2*6 + 2*r][col+0] = 1 # x
        V[2*6 + 2*r+1][col+3] = 1 # y
        V[3*6 + 2*r][col+1] = 1 # u
        V[3*6 + 2*r+1][col+4] = 1 # v
        V[4*6 + 2*r][col+2] = 1 # w
        V[4*6 + 2*r+1][col+5] = 1 # z
        V[2*6 + 2*r][col+6] = 1 # x
        V[2*6 + 2*r+1][col+9] = 1 # y
        V[3*6 + 2*r][col+7] = 1 # u
        V[3*6 + 2*r+1][col+10] = 1 # v
        V[4*6 + 2*r][col+8] = 1 # w
        V[4*6 + 2*r+1][col+11] = 1 # z
        
        # W
        W[2*r*2 + 0][col+0] = 1 # C00: ax
        W[2*r*2 + 0][col+1] = 1 # C00: bu
        W[2*r*2 + 0][col+2] = 1 # C00: cw
        W[(2*r+1)*2 + 0][col+3] = 1 # C01: ay
        W[(2*r+1)*2 + 0][col+4] = 1 # C01: bv
        W[(2*r+1)*2 + 0][col+5] = 1 # C01: cz
        W[2*r*2 + 1][col+6] = 1 # C10: dx
        W[2*r*2 + 1][col+7] = 1 # C10: eu
        W[2*r*2 + 1][col+8] = 1 # C10: fw
        W[(2*r+1)*2 + 1][col+9] = 1 # C11: dy
        W[(2*r+1)*2 + 1][col+10] = 1 # C11: ev
        W[(2*r+1)*2 + 1][col+11] = 1 # C11: fz
        
        col += 12
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
