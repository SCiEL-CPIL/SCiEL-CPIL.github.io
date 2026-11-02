"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Deadline guard: 120s limit, use 110s to be safe
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Decomposition: 1 block of 2x2x1 and 7 blocks of 2x2x2
    # Total rank = 4 + 7*7 = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 53
    V = [[0] * rank for _ in range(m * p)] # 30 x 53
    W = [[0] * rank for _ in range(p * n)] # 30 x 53
    
    col = 0
    
    # Helper to add a block decomposition
    # A is 2x2, B is 2xK, C is Kx2
    # U_block: 4 x R_block, V_block: 2K x R_block, W_block: 2K x R_block
    # Indices:
    # U row: i*2 + j
    # V row: j*K + k
    # W row: k*2 + i
    
    def add_block(k_start, k_end, u_block, v_block, w_block):
        nonlocal col
        R_block = len(u_block[0])
        for c in range(R_block):
            # U
            for i in range(2):
                for j in range(2):
                    U[i*2 + j][col] += u_block[i*2 + j][c]
            # V
            for j in range(2):
                for k in range(k_start, k_end):
                    # V index is j*p + k
                    # v_block index is j*(k_end-k_start) + (k-k_start)
                    V[j*p + k][col] += v_block[j*(k_end-k_start) + (k-k_start)][c]
            # W
            for k in range(k_start, k_end):
                for i in range(2):
                    # W index is k*n + i
                    # w_block index is (k-k_start)*2 + i
                    W[k*n + i][col] += w_block[(k-k_start)*2 + i][c]
            col += 1
        
    # 1. Block for k=0 (2x2x1)
    # Standard multiplication: C[i][k] = sum_j A[i][j] B[j][k]
    # Here k is fixed to 0.
    # Ranks 4: (i,j,k) combinations.
    # U[i,j] = 1 if (i,j) matches
    # V[j,0] = 1 if j matches
    # W[0,i] = 1 if i matches
    
    u_221 = [[0]*4 for _ in range(4)]
    v_221 = [[0]*4 for _ in range(2)] # 2x1 input B part
    w_221 = [[0]*4 for _ in range(2)] # 1x2 output C part
    
    c_idx = 0
    for i in range(2):
        for j in range(2):
            u_221[i*2 + j][c_idx] = 1
            v_221[j][c_idx] = 1
            w_221[0*2 + i][c_idx] = 1 # k=0, so row 0*2+i
            c_idx += 1
            
    add_block(0, 1, u_221, v_221, w_221)
    
    # 2. Seven blocks for k in {1,2}, {3,4}, ..., {13,14}
    # Strassen 2x2x2
    # U (4x7), V (4x7), W (4x7)
    # Standard Strassen factors:
    # U00=1, U01=1, U10=1, U11=-1
    # V00=1, V01=1, V10=1, V11=-1
    # W00=1, W01=1, W10=1, W11=1
    # Wait, let's use the standard symmetric Strassen or just one known set.
    # Common Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # U factors (A side):
    # M1: A00=1, A10=0, A11=1, A01=0 -> [1,0,0,1] (rows 00,01,10,11)
    # M2: A00=0, A10=1, A11=1, A01=0 -> [0,0,1,1]
    # M3: A00=1, A10=0, A11=0, A01=0 -> [1,0,0,0]
    # M4: A00=0, A10=0, A11=1, A01=0 -> [0,0,0,1]
    # M5: A00=1, A10=0, A11=0, A01=1 -> [1,1,0,0]
    # M6: A00=-1, A10=1, A11=0, A01=0 -> [-1,0,1,0]
    # M7: A00=0, A10=0, A11=-1, A01=1 -> [0,1,0,-1]
    
    # V factors (B side):
    # M1: B00=1, B01=0, B10=0, B11=1 -> [1,0,0,1]
    # M2: B00=1, B01=0, B10=0, B11=0 -> [1,0,0,0]
    # M3: B00=0, B01=1, B10=0, B11=-1 -> [0,1,0,-1]
    # M4: B00=-1, B01=0, B10=1, B11=0 -> [-1,0,1,0]
    # M5: B00=0, B01=0, B10=0, B11=1 -> [0,0,0,1]
    # M6: B00=1, B01=1, B10=0, B11=0 -> [1,1,0,0]
    # M7: B00=0, B01=0, B10=1, B11=1 -> [0,0,1,1]
    
    # W factors (C side):
    # C00: M1=1, M4=-1, M5=1, M7=1 -> [1,0,0,-1,1,0,1]
    # C01: M1=0, M4=0, M5=1, M7=0 -> [0,0,0,0,1,0,0] wait, C01 = M3+M5.
    # Let's re-verify C formulas.
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # W rows correspond to C00, C01, C10, C11 (indices 0,1,2,3 in 2x2 block)
    # W0 (C00): [1, 0, 0, -1, 1, 0, 1]
    # W1 (C01): [0, 0, 1, 0, 1, 0, 0]
    # W2 (C10): [0, 1, 0, 1, 0, 0, 0]
    # W3 (C11): [1, 0, -1, 0, 0, 1, 1]
    
    u_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    v_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    w_strassen = [
        [1, 0, 0, 1, 0, 0, 0], # C00? No.
        # Let's map carefully.
        # U rows: 00, 01, 10, 11
        # V rows: 00, 01, 10, 11
        # W rows: 00, 01, 10, 11
        
        # M1: U[0]=1, U[3]=1. V[0]=1, V[3]=1.
        # M2: U[2]=1, U[3]=1. V[0]=1.
        # M3: U[0]=1. V[1]=1, V[3]=-1.
        # M4: U[3]=1. V[0]=-1, V[2]=1.
        # M5: U[0]=1, U[1]=1. V[3]=1.
        # M6: U[0]=-1, U[2]=1. V[0]=1, V[1]=1.
        # M7: U[1]=1, U[3]=-1. V[2]=1, V[3]=1.
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 + M7
        
        # W[0] (C00): M1(1), M2(0), M3(0), M4(-1), M5(1), M6(0), M7(1)
        # W[1] (C01): M1(0), M2(0), M3(1), M4(0), M5(1), M6(0), M7(0)
        # W[2] (C10): M1(0), M2(1), M3(0), M4(1), M5(0), M6(0), M7(0)
        # W[3] (C11): M1(1), M2(0), M3(-1), M4(0), M5(0), M6(1), M7(1)
    ]
    
    w_strassen = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]
    
    for b in range(1, 8):
        k_start = 2 * b - 1
        k_end = 2 * b + 1
        if time.time() > deadline:
             break
        add_block(k_start, k_end, u_strassen, v_strassen, w_strassen)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
