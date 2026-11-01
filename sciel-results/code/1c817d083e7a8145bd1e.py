"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    col = 0
    
    def add_strassen_222(u_offset, v_col_offset, w_col_offset):
        nonlocal col
        # u_offset: base index for U rows (i*m + j)
        # v_col_offset: base index for V cols (j*p + l) -> l is 0,1
        # w_col_offset: base index for W cols (l*n + i) -> l is 0,1
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[u_offset + 0][col] = 1
        U[u_offset + 1][col] = 1
        V[v_col_offset + 0][col] = 1
        V[v_col_offset + 1][col] = 1
        W[w_col_offset + 0][col] = 1  # Z00
        W[w_col_offset + 3][col] = 1  # Z11 (l=1, i=1 -> 1*2+1=3)
        col += 1
        
        # M2 = (X10 + X11)Y00
        U[u_offset + 3][col] = 1  # X10 is i=1, j=0 -> 1*2+0=3? No.
        # Wait, U index is i*m + j. m=3.
        # X is 2x2.
        # X00: i=0, j=0 -> 0*3+0=0
        # X10: i=1, j=0 -> 1*3+0=3
        # X11: i=1, j=1 -> 1*3+1=4
        # X01: i=0, j=1 -> 0*3+1=1
        # Correction:
        # M1: X00(0), X11(4).
        # M2: X10(3), X11(4).
        # M3: X00(0).
        # M4: X11(4).
        # M5: X00(0), X10(3).
        # M6: X11(4), X00(0).
        # M7: X01(1), X11(4).
        
        # Let's re-verify indices for U (A part)
        # A is 2x3.
        # A11 is A[0:2, 0:2].
        # A11[0,0] -> U[0*3+0] = U[0]
        # A11[0,1] -> U[0*3+1] = U[1]
        # A11[1,0] -> U[1*3+0] = U[3]
        # A11[1,1] -> U[1*3+1] = U[4]
        
        # So:
        # M1: U[0]=1, U[4]=1.
        # M2: U[3]=1, U[4]=1.
        # M3: U[0]=1.
        # M4: U[4]=1.
        # M5: U[0]=1, U[3]=1.
        # M6: U[4]=1, U[0]=-1.
        # M7: U[1]=1, U[4]=-1.
        
        # V indices for B part (3x2 block)
        # B is 3x5.
        # B1 is B[0:3, 0:2].
        # B1[0,0] -> V[0*5+0] = V[0]
        # B1[0,1] -> V[0*5+1] = V[1]
        # B1[1,0] -> V[1*5+0] = V[5]
        # B1[1,1] -> V[1*5+1] = V[6]
        # B1[2,0] -> V[2*5+0] = V[10]
        # B1[2,1] -> V[2*5+1] = V[11]
        
        # Y00 -> V[0]
        # Y01 -> V[1]
        # Y10 -> V[5]
        # Y11 -> V[6]
        
        # W indices for C part (2x2 block)
        # C is 2x5.
        # C1 is C[0:2, 0:2].
        # C1[0,0] -> W[0*2+0] = W[0]
        # C1[0,1] -> W[1*2+0] = W[2]
        # C1[1,0] -> W[0*2+1] = W[1]
        # C1[1,1] -> W[1*2+1] = W[3]
        
        # Z00 -> W[0]
        # Z01 -> W[2]
        # Z10 -> W[1]
        # Z11 -> W[3]
        
        # Re-do M1
        U[u_offset + 0][col] = 1
        U[u_offset + 4][col] = 1
        V[v_col_offset + 0][col] = 1
        V[v_col_offset + 6][col] = 1
        W[w_col_offset + 0][col] = 1
        W[w_col_offset + 3][col] = 1
        col += 1
        
        # M2
        U[u_offset + 3][col] = 1
        U[u_offset + 4][col] = 1
        V[v_col_offset + 0][col] = 1
        W[w_col_offset + 1][col] = 1 # Z10
        col += 1
        
        # M3
        U[u_offset + 0][col] = 1
        V[v_col_offset + 5][col] = 1 # Y10
        V[v_col_offset + 6][col] = -1 # Y11
        W[w_col_offset + 2][col] = 1 # Z01
        col += 1
        
        # M4
        U[u_offset + 4][col] = 1
        V[v_col_offset + 1][col] = 1 # Y01
        V[v_col_offset + 0][col] = -1 # Y00
        W[w_col_offset + 1][col] = 1 # Z10
        W[w_col_offset + 3][col] = 1 # Z11
        col += 1
        
        # M5
        U[u_offset + 0][col] = 1
        U[u_offset + 3][col] = 1
        V[v_col_offset + 6][col] = 1 # Y11
        W[w_col_offset + 0][col] = -1 # Z00
        W[w_col_offset + 2][col] = 1 # Z01
        col += 1
        
        # M6
        U[u_offset + 4][col] = 1
        U[u_offset + 0][col] = -1
        V[v_col_offset + 0][col] = 1 # Y00
        V[v_col_offset + 5][col] = 1 # Y10
        W[w_col_offset + 3][col] = 1 # Z11
        col += 1
        
        # M7
        U[u_offset + 1][col] = 1
        U[u_offset + 4][col] = -1
        V[v_col_offset + 6][col] = 1 # Y11
        V[v_col_offset + 5][col] = 1 # Y10
        W[w_col_offset + 0][col] = 1 # Z00
        W[w_col_offset + 2][col] = -1 # Z01
        col += 1

    def add_schoolbook_212(u_offset, v_col_offset, w_col_offset):
        nonlocal col
        # X is 2x1 (A12). Y is 1x2 (B12).
        # X00 -> U[u_offset + 0] (i=0, j=2 -> 0*3+2=2)
        # X10 -> U[u_offset + 1] (i=1, j=2 -> 1*3+2=5)
        # Wait, u_offset is base for A.
        # A12 is A[:, 2].
        # A[0,2] -> U[0*3+2] = U[2]
        # A[1,2] -> U[1*3+2] = U[5]
        
        # Y is B[2, 0:2].
        # Y00 -> V[v_col_offset + 10] (j=2, l=0 -> 2*5+0=10)
        # Y01 -> V[v_col_offset + 11] (j=2, l=1 -> 2*5+1=11)
        
        # Z is C[:, 0:2].
        # Z00 -> W[w_col_offset + 0]
        # Z01 -> W[w_col_offset + 2]
        # Z10 -> W[w_col_offset + 1]
        # Z11 -> W[w_col_offset + 3]
        
        # Term 1: Z00 = X00 * Y00
        U[u_offset + 2][col] = 1
        V[v_col_offset + 10][col] = 1
        W[w_col_offset + 0][col] = 1
        col += 1
        
        # Term 2: Z01 = X00 * Y01
        U[u_offset + 2][col] = 1
        V[v_col_offset + 11][col] = 1
        W[w_col_offset + 2][col] = 1
        col += 1
        
        # Term 3: Z10 = X10 * Y00
        U[u_offset + 5][col] = 1
        V[v_col_offset + 10][col] = 1
        W[w_col_offset + 1][col] = 1
        col += 1
        
        # Term 4: Z11 = X10 * Y01
        U[u_offset + 5][col] = 1
        V[v_col_offset + 11][col] = 1
        W[w_col_offset + 3][col] = 1
        col += 1

    def add_schoolbook_231(u_offset, v_col_offset, w_col_offset):
        nonlocal col
        # X is 2x3 (A). Y is 3x1 (B3).
        # Y is B[:, 4].
        # Y0 -> V[v_col_offset + 4]
        # Y1 -> V[v_col_offset + 9]
        # Y2 -> V[v_col_offset + 14]
        
        # Z is C[:, 4].
        # Z0 -> W[w_col_offset + 8] (l=4, i=0 -> 4*2+0=8)
        # Z1 -> W[w_col_offset + 9] (l=4, i=1 -> 4*2+1=9)
        
        for i in range(2):
            for j in range(3):
                U[u_offset + i*m + j][col] = 1
                V[v_col_offset + j*p + 4][col] = 1
                W[w_col_offset + 4*n + i][col] = 1
                col += 1

    # Block 1: A11 * B11 (Strassen) + A12 * B12 (Schoolbook)
    # B11 is B[:, 0:2]. B12 is B[:, 0:2] but row 2?
    # Wait, B1 is 3x2.
    # Strassen uses B1[0:2, 0:2].
    # Schoolbook uses B1[2, 0:2].
    # So v_col_offset is 0 for both.
    # w_col_offset is 0 for both.
    
    add_strassen_222(0, 0, 0)
    add_schoolbook_212(0, 0, 0)
    
    # Block 2: A11 * B21 (Strassen) + A12 * B22 (Schoolbook)
    # B2 is B[:, 2:4].
    # v_col_offset is 2.
    # w_col_offset is 2.
    
    add_strassen_222(0, 2, 2)
    add_schoolbook_212(0, 2, 2)
    
    # Block 3: A * B3 (Schoolbook)
    # B3 is B[:, 4:5].
    # v_col_offset is 4.
    # w_col_offset is 4.
    
    add_schoolbook_231(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
