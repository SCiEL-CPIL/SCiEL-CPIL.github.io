"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    # U, V are 4x7, W is 4x7 (flattened k*n+i)
    # Indices: i,j in 0..1
    # U[i*2+j, r]
    # V[j*2+k, r]
    # W[k*2+i, r]
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    # V: (0,0)=1, (1,1)=1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    # W: C00 += 1, C11 += 1
    # C00 is i=0, k=0 -> W[0*2+0] = W[0]
    # C11 is i=1, k=1 -> W[1*2+1] = W[3]
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2: (A10+A11)B00
    # U: (1,0)=1, (1,1)=1
    U_222[2, 1] = 1; U_222[3, 1] = 1
    # V: (0,0)=1
    V_222[0, 1] = 1
    # W: C10 += 1, C11 -= 1
    # C10 is i=1, k=0 -> W[0*2+1] = W[1]
    # C11 is i=1, k=1 -> W[3]
    W_222[1, 1] = 1; W_222[3, 1] = -1
    
    # M3: A00(B01-B11)
    # U: (0,0)=1
    U_222[0, 2] = 1
    # V: (0,1)=1, (1,1)=-1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    # W: C01 += 1, C11 += 1
    # C01 is i=0, k=1 -> W[1*2+0] = W[2]
    # C11 is i=1, k=1 -> W[3]
    W_222[2, 2] = 1; W_222[3, 2] = 1
    
    # M4: A11(B10-B00)
    # U: (1,1)=1
    U_222[3, 3] = 1
    # V: (1,0)=1, (0,0)=-1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    # W: C00 += 1, C10 += 1
    # C00 is i=0, k=0 -> W[0]
    # C10 is i=1, k=0 -> W[1]
    W_222[0, 3] = 1; W_222[1, 3] = 1
    
    # M5: (A00+A01)B11
    # U: (0,0)=1, (0,1)=1
    U_222[0, 4] = 1; U_222[1, 4] = 1
    # V: (1,1)=1
    V_222[3, 4] = 1
    # W: C00 -= 1, C01 += 1
    # C00 is i=0, k=0 -> W[0]
    # C01 is i=0, k=1 -> W[2]
    W_222[0, 4] = -1; W_222[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: (1,0)=1, (0,0)=-1
    U_222[2, 5] = 1; U_222[0, 5] = -1
    # V: (0,0)=1, (0,1)=1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    # W: C11 += 1
    # C11 is i=1, k=1 -> W[3]
    W_222[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: (0,1)=1, (1,1)=-1
    U_222[1, 6] = 1; U_222[3, 6] = -1
    # V: (1,0)=1, (1,1)=1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    # W: C00 += 1
    # C00 is i=0, k=0 -> W[0]
    W_222[0, 6] = 1

    # Construct 4x4x4 decomposition
    # Rank 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Blocks: 00, 01, 10, 11
    # M1: A00+A11, B00+B11 -> C00, C11
    # M2: A10+A11, B00 -> C10, C11
    # M3: A00, B01-B11 -> C01, C11
    # M4: A11, B10-B00 -> C00, C10
    # M5: A00+A01, B11 -> C00, C01
    # M6: A10-A00, B00+B01 -> C11
    # M7: A01-A11, B10+B11 -> C00
    
    def add_block(U, V, W, u_blocks, v_blocks, w_blocks, sign, offset):
        # u_blocks: list of (row_start, col_start, sign_u)
        # Actually, U is 4x7 for 2x2.
        # We need to place U_222 into U_444.
        # U_444 is 16x49.
        # Row index i*4+j.
        # For block (bi, bj), rows are (bi*2 + i)*4 + (bj*2 + j).
        
        for r in range(7):
            col = offset + r
            # U
            for (bi, bj, su) in u_blocks:
                for i in range(2):
                    for j in range(2):
                        val = U_222[i*2+j, r] * su
                        if val != 0:
                            row = (bi*2 + i)*4 + (bj*2 + j)
                            U_444[row, col] += val
                            
            # V
            for (bj, bk, sv) in v_blocks:
                for j in range(2):
                    for k in range(2):
                        val = V_222[j*2+k, r] * sv
                        if val != 0:
                            row = (bj*2 + j)*4 + (bk*2 + k)
                            V_444[row, col] += val
                            
            # W
            for (bk, bi, sw) in w_blocks:
                for k in range(2):
                    for i in range(2):
                        val = W_222[k*2+i, r] * sw
                        if val != 0:
                            row = (bk*2 + k)*4 + (bi*2 + i)
                            W_444[row, col] += val

    # M1
    u1 = [(0,0,1), (1,1,1)]
    v1 = [(0,0,1), (1,1,1)]
    w1 = [(0,0,1), (1,1,1)] # C00, C11
    add_block(U_444, V_444, W_444, u1, v1, w1, 1, 0)
    
    # M2
    u2 = [(1,0,1), (1,1,1)]
    v2 = [(0,0,1)]
    w2 = [(0,1,1), (1,1,-1)] # C10, C11
    add_block(U_444, V_444, W_444, u2, v2, w2, 1, 7)
    
    # M3
    u3 = [(0,0,1)]
    v3 = [(0,1,1), (1,1,-1)]
    w3 = [(1,0,1), (1,1,1)] # C01, C11
    add_block(U_444, V_444, W_444, u3, v3, w3, 1, 14)
    
    # M4
    u4 = [(1,1,1)]
    v4 = [(1,0,1), (0,0,-1)]
    w4 = [(0,0,1), (0,1,1)] # C00, C10
    add_block(U_444, V_444, W_444, u4, v4, w4, 1, 21)
    
    # M5
    u5 = [(0,0,1), (0,1,1)]
    v5 = [(1,1,1)]
    w5 = [(0,0,-1), (1,0,1)] # C00, C01
    add_block(U_444, V_444, W_444, u5, v5, w5, 1, 28)
    
    # M6
    u6 = [(1,0,1), (0,0,-1)]
    v6 = [(0,0,1), (0,1,1)]
    w6 = [(1,1,1)] # C11
    add_block(U_444, V_444, W_444, u6, v6, w6, 1, 35)
    
    # M7
    u7 = [(0,1,1), (1,1,-1)]
    v7 = [(1,0,1), (1,1,1)]
    w7 = [(0,0,1)] # C00
    add_block(U_444, V_444, W_444, u7, v7, w7, 1, 42)

    # Add 5th column: 4x4x1
    # Rank 16
    U_445 = np.zeros((16, 65))
    V_445 = np.zeros((20, 65))
    W_445 = np.zeros((20, 65))
    
    # Copy 4x4x4 part
    U_445[:, :49] = U_444
    V_445[:, :49] = V_444
    W_445[:, :49] = W_444
    
    # Add 16 terms for k=4
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            U_445[i*4+j, r] = 1
            V_445[j*5+4, r] = 1
            W_445[4*4+i, r] = 1 # k=4, i -> 4*4+i

    return {"U": U_445.tolist(), "V": V_445.tolist(), "W": W_445.tolist()}
# EVOLVE-BLOCK-END

