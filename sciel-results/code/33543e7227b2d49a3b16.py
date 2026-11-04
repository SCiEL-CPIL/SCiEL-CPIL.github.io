"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _strassen_4x4_4x4():
    """
    Rank-49 decomposition for 4x4 * 4x4 -> 4x4 using Strassen squared.
    
    Returns (U, V, W) where:
      U: (16, 49) - A[i,j] with row = i*4+j
      V: (16, 49) - B[j,k] with row = j*4+k
      W: (16, 49) - C[i,k] with row = k*4+i (trace convention)
    """
    sU, sV, sW = _strassen_2x2()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for t_outer in range(7):
        for t_inner in range(7):
            col = t_outer * 7 + t_inner
            
            # U: outer coefficient * inner coefficient
            for ai in range(2):
                for bj in range(2):
                    c_outer = sU[ai*2 + bj, t_outer]
                    if c_outer == 0:
                        continue
                    for ai2 in range(2):
                        for bj2 in range(2):
                            c_inner = sU[ai2*2 + bj2, t_inner]
                            if c_inner == 0:
                                continue
                            row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[row, col] += c_outer * c_inner
            
            # V: outer coefficient * inner coefficient
            for bj in range(2):
                for ck in range(2):
                    c_outer = sV[bj*2 + ck, t_outer]
                    if c_outer == 0:
                        continue
                    for bj2 in range(2):
                        for ck2 in range(2):
                            c_inner = sV[bj2*2 + ck2, t_inner]
                            if c_inner == 0:
                                continue
                            row = (2*bj + bj2) * 4 + (2*ck + ck2)
                            V[row, col] += c_outer * c_inner
            
            # W: outer coefficient * inner coefficient
            for ai in range(2):
                for ck in range(2):
                    c_outer = sW[ck*2 + ai, t_outer]
                    if c_outer == 0:
                        continue
                    for ai2 in range(2):
                        for ck2 in range(2):
                            c_inner = sW[ck2*2 + ai2, t_inner]
                            if c_inner == 0:
                                continue
                            row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[row, col] += c_outer * c_inner
    
    return U, V, W


def _strassen_4x4_4x2():
    """
    Rank-28 decomposition for 4x4 * 4x2 -> 4x2 using Strassen on 2x2 blocks.
    
    A is 4x4 (2x2 blocks of 2x2), B is 4x2 (2x1 blocks of 2x1... 
    actually B is 4x2, so in 2x2 blocks it's 2 rows x 1 col of 2x2 blocks).
    C is 4x2 (2x1 blocks of 2x2).
    
    Returns (U, V, W) where:
      U: (16, 28) - A[i,j] with row = i*4+j
      V: (8, 28)  - B[j,k] with row = j*2+k
      W: (8, 28)  - C[i,k] with row = k*4+i (trace convention: W[k*n+i], n=4)
    """
    sU, sV, sW = _strassen_2x2()
    
    U = np.zeros((16, 28), dtype=int)
    V = np.zeros((8, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    # Outer: 2x2 (A blocks) * 2x1 (B blocks) -> 2x1 (C blocks)
    # Schoolbook outer: 4 terms for (ai, bj) pairs
    # Each term: A[ai][bj] * B[bj][0] -> C[ai][0]
    # Inner: 2x2 * 2x2 -> 2x2 (Strassen, rank 7)
    
    for ai in range(2):
        for bj in range(2):
            for t in range(7):
                col = (ai * 2 + bj) * 7 + t
                
                # U: A[ai][bj] block, inner Strassen
                for ai2 in range(2):
                    for bj2 in range(2):
                        c_inner = sU[ai2*2 + bj2, t]
                        if c_inner == 0:
                            continue
                        row = (2*ai + ai2) * 4 + (2*bj + bj2)
                        U[row, col] += c_inner
                
                # V: B[bj][0] block, inner Strassen
                # B is 4x2, so B[bj_block][0] is a 2x2 block
                # B row index: (2*bj + bj2)*2 + ck2 where ck2 is the column within the 2x2 block
                # But B only has 2 columns total, so ck2 ranges 0,1
                for bj2 in range(2):
                    for ck2 in range(2):
                        c_inner = sV[bj2*2 + ck2, t]
                        if c_inner == 0:
                            continue
                        row = (2*bj + bj2) * 2 + ck2
                        V[row, col] += c_inner
                
                # W: C[ai][0] block, inner Strassen
                # C is 4x2, so C[ai_block][0] is a 2x2 block
                # C row index: (2*ck + ck2)*4 + (2*ai + ai2) where ck is the block col (always 0)
                # Wait, trace convention: W[k*n+i] where n=4 (number of rows of C)
                # C[i,k] with i in 0..3, k in 0..1
                # W row = k*4 + i
                # For C[ai_block][0_block]: i = 2*ai+ai2, k = 2*0+ck2 = ck2
                for ai2 in range(2):
                    for ck2 in range(2):
                        c_inner = sW[ck2*2 + ai2, t]
                        if c_inner == 0:
                            continue
                        row = ck2 * 4 + (2*ai + ai2)
                        W[row, col] += c_inner
    
    return U, V, W


import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: 
    # Part 1: 4x4 * 4x4 -> 4x4 (first 4 columns of B/C) using rank-49 Strassen squared
    # Part 2: 4x4 * 4x2 -> 4x2 (last 2 columns of B/C) using rank-28 Strassen
    # Total rank: 49 + 28 = 77
    
    U1, V1, W1 = _strassen_4x4_4x4()
    U2, V2, W2 = _strassen_4x4_4x2()
    
    total_rank = 49 + 28
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: columns 0-3 of B, columns 0-3 of C
    # U1 is (16, 49)
    U[:, :49] = U1
    
    # V1 row = j*4+k for j,k in 0..3. Map to V row = j*6+k for k in 0..3
    for j in range(4):
        for k in range(4):
            V[j*6 + k, :49] = V1[j*4 + k, :49]
    
    # W1 row = k*4+i for k,i in 0..3. This maps directly to W rows 0-15
    W[:16, :49] = W1
    
    # Part 2: columns 4-5 of B, columns 4-5 of C
    # U2 is (16, 28), V2 is (8, 28), W2 is (8, 28)
    # V2 row = j*2+k for 4x2 B, where j in 0..3, k in 0..1
    # We need to map to V row = j*6+k for 4x6 B, where k in 4..5
    # So V2 row (j*2+k) maps to V row (j*6 + (k+4))
    
    U[:, 49:] = U2
    
    for j in range(4):
        for k in range(2):
            V[j*6 + (k+4), 49:] = V2[j*2 + k, :]
    
    # W2 row = k*4+i for 4x2 C, where k in 0..1, i in 0..3
    # We need W row = k*4+i for 4x6 C, where k in 4..5, i in 0..3
    # So W2 row (k*4+i) maps to W row ((k+4)*4 + i)
    
    for k in range(2):
        for i in range(4):
            W[(k+4)*4 + i, 49:] = W2[k*4 + i, :]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Should not happen for this fixed-size problem, but guard anyway
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
