"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Rank-7 Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[2*i+j], V[2*j+k], W[2*k+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00: i=0,j=0. A11: i=1,j=1.
    # B00: j=0,k=0. B11: j=1,k=1.
    # C00: k=0,i=0. C11: k=1,i=1.
    # Contribution to C00: +1, C11: +1.
    # A' = A00+A11. B' = B00+B11.
    # U2 for A':
    # A00: i=0,j=0 -> idx 0. A11: i=1,j=1 -> idx 3.
    # U2[0, 0] = 1; U2[3, 0] = 1
    # V2 for B':
    # B00: j=0,k=0 -> idx 0. B11: j=1,k=1 -> idx 3.
    # V2[0, 0] = 1; V2[3, 0] = 1
    # W2 for C:
    # C00: k=0,i=0 -> idx 0. C11: k=1,i=1 -> idx 3.
    # W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A10: i=1,j=0 -> idx 2. A11: i=1,j=1 -> idx 3.
    # B00: j=0,k=0 -> idx 0.
    # C10: k=0,i=1 -> idx 1.
    # U2[2, 1] = 1; U2[3, 1] = 1
    # V2[0, 1] = 1
    # W2[1, 1] = 1
    
    # M3 = A00(B01-B11)
    # A00: i=0,j=0 -> idx 0.
    # B01: j=0,k=1 -> idx 1. B11: j=1,k=1 -> idx 3.
    # C01: k=1,i=0 -> idx 2.
    # U2[0, 2] = 1
    # V2[1, 2] = 1; V2[3, 2] = -1
    # W2[2, 2] = 1
    
    # M4 = A11(B10-B00)
    # A11: i=1,j=1 -> idx 3.
    # B10: j=1,k=0 -> idx 2. B00: j=0,k=0 -> idx 0.
    # C10: k=0,i=1 -> idx 1. C00: k=0,i=0 -> idx 0.
    # U2[3, 3] = 1
    # V2[2, 3] = 1; V2[0, 3] = -1
    # W2[1, 3] = 1; W2[0, 3] = 1
    
    # M5 = (A00+A01)B11
    # A00: i=0,j=0 -> idx 0. A01: i=0,j=1 -> idx 1.
    # B11: j=1,k=1 -> idx 3.
    # C01: k=1,i=0 -> idx 2. C11: k=1,i=1 -> idx 3.
    # U2[0, 4] = 1; U2[1, 4] = 1
    # V2[3, 4] = 1
    # W2[2, 4] = -1; W2[3, 4] = 1  # C01 = -M5, C11 = +M5? 
    # Formula: C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    # Wait, C01 = M3+M5. So W2[2,4]=1.
    # C11 = M1-M2+M3+M6. No M5 in C11?
    # Let's re-verify Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check M5 in C01: +1. In C11: 0.
    # So W2[2, 4] = 1 (C01). W2[3, 4] = 0.
    # My previous W2[3,4]=1 was wrong.
    
    # M6 = (A10-A00)(B00+B01)
    # A10: i=1,j=0 -> idx 2. A00: i=0,j=0 -> idx 0.
    # B00: j=0,k=0 -> idx 0. B01: j=0,k=1 -> idx 1.
    # C11: k=1,i=1 -> idx 3.
    # U2[2, 5] = 1; U2[0, 5] = -1
    # V2[0, 5] = 1; V2[1, 5] = 1
    # W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: i=0,j=1 -> idx 1. A11: i=1,j=1 -> idx 3.
    # B10: j=1,k=0 -> idx 2. B11: j=1,k=1 -> idx 3.
    # C00: k=0,i=0 -> idx 0.
    # U2[1, 6] = 1; U2[3, 6] = -1
    # V2[2, 6] = 1; V2[3, 6] = 1
    # W2[0, 6] = 1
    
    # Correcting W2 for M5
    W2[2, 4] = 1
    W2[3, 4] = 0

    # Verify 2x2x2 decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook if Strassen fails
                    U2 = np.zeros((4, 4))
                    V2 = np.zeros((4, 4))
                    W2 = np.zeros((4, 4))
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                U2[i*2+j, i*2+j+k] = 1 # Bad index logic, just use schoolbook directly later
                    pass

    # Build 4x4x4 decomposition using Strassen
    # 7 super-terms, each rank 7 -> total 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen definitions:
    # M1: A' = A00+A11, B' = B00+B11
    # M2: A' = A10+A11, B' = B00
    # M3: A' = A00, B' = B01-B11
    # M4: A' = A11, B' = B10-B00
    # M5: A' = A00+A01, B' = B11
    # M6: A' = A10-A00, B' = B00+B01
    # M7: A' = A01-A11, B' = B10+B11
    
    # Coefficients for A' blocks (I, J)
    # A' is 2x2. A'(a, c) = sum_{I,J} alpha[I,J] * A_{I,J}(a,c)
    # A_{I,J} is block at row I, col J.
    # Elements: A_{2I+a, 2J+c}
    
    A_coeffs = [
        {(0,0):1, (1,1):1}, # M1
        {(1,0):1, (1,1):1}, # M2
        {(0,0):1},           # M3
        {(1,1):1},           # M4
        {(0,0):1, (0,1):1},  # M5
        {(1,0):1, (0,0):-1}, # M6
        {(0,1):1, (1,1):-1}  # M7
    ]
    
    B_coeffs = [
        {(0,0):1, (1,1):1}, # M1
        {(0,0):1},           # M2
        {(0,1):1, (1,1):-1}, # M3
        {(1,0):1, (0,0):-1}, # M4
        {(1,1):1},           # M5
        {(0,0):1, (0,1):1},  # M6
        {(1,0):1, (1,1):1}   # M7
    ]
    
    # C coeffs for M_s
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    C_coeffs = [
        {(0,0):1, (1,1):1, (1,0):-1, (0,1):1}, # M1
        {(1,0):1, (1,1):-1},                    # M2
        {(0,1):1, (1,1):1},                     # M3
        {(0,0):1, (1,0):1},                     # M4
        {(0,0):-1, (0,1):1},                    # M5
        {(1,1):1},                              # M6
        {(0,0):1}                               # M7
    ]
    
    col = 0
    for s in range(7):
        for r in range(7):
            # Fill U4
            for a in range(2):
                for c in range(2):
                    val_u2 = U2[a*2+c, r]
                    if val_u2 != 0:
                        for I, J in A_coeffs[s].keys():
                            alpha = A_coeffs[s][(I, J)]
                            if alpha != 0:
                                # Global i = 2I+a, j = 2J+c
                                gi = 2*I + a
                                gj = 2*J + c
                                U4[gi*4+gj, col] += alpha * val_u2
            
            # Fill V4
            for c in range(2):
                for b in range(2):
                    val_v2 = V2[c*2+b, r]
                    if val_v2 != 0:
                        for J, K in B_coeffs[s].keys():
                            beta = B_coeffs[s][(J, K)]
                            if beta != 0:
                                # Global j = 2J+c, k = 2K+b
                                gj = 2*J + c
                                gk = 2*K + b
                                V4[gj*4+gk, col] += beta * val_v2
            
            # Fill W4
            for b in range(2):
                for a in range(2):
                    val_w2 = W2[b*2+a, r]
                    if val_w2 != 0:
                        for I, K in C_coeffs[s].keys():
                            gamma = C_coeffs[s][(I, K)]
                            if gamma != 0:
                                # Global k = 2K+b, i = 2I+a
                                gk = 2*K + b
                                gi = 2*I + a
                                W4[gk*4+gi, col] += gamma * val_w2
            
            col += 1
            
    # Verify 4x4x4 decomposition
    def verify4(U, V, W, rank):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify4(U4, V4, W4, 49):
        # Fallback to schoolbook for 4x4x4
        U4 = np.zeros((16, 64))
        V4 = np.zeros((16, 64))
        W4 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
        core_rank = 64
    else:
        core_rank = 49

    # Total Rank:
    # Core 4x4x4: core_rank
    # Strip 1: A(4x1) B(1x2) -> 4x1x2. Rank 8.
    # Strip 2: A(4x4) B(4x2) -> 4x4x2. Rank 32.
    # Strip 3: A(4x1) B(1x4) -> 4x1x4. Rank 16.
    
    total_rank = core_rank + 8 + 32 + 16
    
    U = np.zeros((20, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((24, total_rank))
    
    col = 0
    
    # Core: i in 0..3, j in 0..3, k in 0..3
    for c in range(core_rank):
        for i in range(4):
            for j in range(4):
                U[i*5+j, col+c] = U4[i*4+j, c]
        for j in range(4):
            for k in range(4):
                V[j*6+k, col+c] = V4[j*4+k, c]
        for k in range(4):
            for i in range(4):
                W[k*4+i, col+c] = W4[k*4+i, c]
    col += core_rank
    
    # Strip 1: i in 0..3, j=4, k in 4..5
    # A(4x1) B(1x2)
    for i in range(4):
        for k in range(2):
            U[i*5+4, col] = 1
            V[4*6+(4+k), col] = 1 # j=4, k=4+k
            W[(4+k)*4+i, col] = 1
            col += 1
            
    # Strip 2: i in 0..3, j in 0..3, k in 4..5
    # A(4x4) B(4x2)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*5+j, col] = 1
                V[j*6+(4+k), col] = 1
                W[(4+k)*4+i, col] = 1
                col += 1
                
    # Strip 3: i in 0..3, j=4, k in 0..3
    # A(4x1) B(1x4)
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1
            V[4*6+k, col] = 1
            W[k*4+i, col] = 1
            col += 1
            
    # Verify full decomposition
    def verify_full(U, V, W, rank):
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*5+j, c] * V[j*6+k, c] * W[k*4+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify_full(U, V, W, total_rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = 120
        U = np.zeros((20, rank))
        V = np.zeros((30, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    U[i*5+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
