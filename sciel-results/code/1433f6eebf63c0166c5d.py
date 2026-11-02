import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Decompose 3x3x3 using schoolbook (Rank 27)
    # U: 9x27, V: 9x27, W: 9x27
    U_333, V_333, W_333 = _schoolbook(3, 3, 3)
    
    # 2. Decompose 3x2x3
    # Split n=3 into 2+1
    # Part A: 2x2x3 (Rank 11)
    #   Split p=3 into 2+1
    #   A1: 2x2x2 (Strassen, Rank 7)
    #   A2: 2x2x1 (Schoolbook, Rank 4)
    # Part B: 1x2x3 (Schoolbook, Rank 6)
    
    # 2A: 2x2x2 Strassen
    U_222, V_222, W_222 = _strassen_222()
    
    # 2A2: 2x2x1 Schoolbook
    U_221, V_221, W_221 = _schoolbook(2, 2, 1)
    
    # Combine 2A1 and 2A2 to get 2x2x3
    # U: 4x(7+4)=4x11
    # V: 4x11 (Note: V for 2x2x2 uses cols 0,1 of B. V for 2x2x1 uses col 2 of B)
    # Wait, V index is j*p + k.
    # For 2x2x2, p=2. V rows 0..3.
    # For 2x2x1, p=1. V rows 0..3.
    # When combining for p=3, V has 6 rows.
    # Rows 0..3 correspond to k=0,1. Rows 4..5 correspond to k=2.
    # So V_222 goes into rows 0..3. V_221 goes into rows 4..5?
    # No, V_221 is for $B_{j, k}$ where $k$ is the single column.
    # In the combined 2x2x3, the second part corresponds to $k=2$.
    # So V_221's rows (which are for $k=0$ in its local frame) map to $k=2$ in global.
    
    U_223 = np.hstack([U_222, U_221])
    V_223 = np.zeros((4, 11))
    W_223 = np.hstack([W_222, W_221])
    
    # Map V_222 (4x7) to V_223 rows 0..3
    V_223[0:4, 0:7] = V_222
    # Map V_221 (4x4) to V_223 rows 4..5?
    # V_221 rows are j*1 + 0 = j. So rows 0,1,2,3.
    # In global 2x2x3, these correspond to j=0,1 and k=2.
    # Global index: j*3 + 2.
    # j=0 -> 2. j=1 -> 5.
    # So V_221[0, :] -> V_223[2, 7:11]
    # V_221[1, :] -> V_223[5, 7:11]
    # V_221[2, :] -> V_223[0+? No, j=0,1 only for 2x2.
    # Wait, 2x2x1 has n=2, m=2, p=1.
    # U is 4x4. V is 4x4. W is 2x4.
    # V rows: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) in local j,k?
    # No, V index is j*m + k? No, V index is j*p + k.
    # For p=1, V index is j*1 + 0 = j.
    # So V_221 has 4 rows? No, m=2, p=1. V shape (2*1, r) = (2, r).
    # Ah, V shape is (m*p, r).
    # For 2x2x2: V is (2*2, 7) = (4, 7).
    # For 2x2x1: V is (2*1, 4) = (2, 4).
    
    # Correct V_223 construction:
    V_223 = np.zeros((6, 11)) # m=2, p=3 -> 6 rows
    # V_222 (4x7) maps to k=0,1.
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # Global indices for k=0,1:
    # (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4.
    V_223[0, 0:7] = V_222[0, :]
    V_223[1, 0:7] = V_222[1, :]
    V_223[3, 0:7] = V_222[2, :]
    V_223[4, 0:7] = V_222[3, :]
    
    # V_221 (2x4) maps to k=2.
    # Local indices: 0:(0,0), 1:(1,0).
    # Global indices for k=2:
    # (0,2)->2, (1,2)->5.
    V_223[2, 7:11] = V_221[0, :]
    V_223[5, 7:11] = V_221[1, :]
    
    # 2B: 1x2x3 Schoolbook
    U_123, V_123, W_123 = _schoolbook(1, 2, 3)
    # U: (1*2, 6) = (2, 6)
    # V: (2*3, 6) = (6, 6)
    # W: (3*1, 6) = (3, 6)
    
    # Combine 2A (2x2x3) and 2B (1x2x3) to get 3x2x3
    # U: 4x11 + 2x6 -> 6x17? No, U shape is (n*m, r).
    # For 2x2x3, U is (4, 11).
    # For 1x2x3, U is (2, 6).
    # Total U for 3x2x3 is (6, 17).
    # Rows 0..3 for n=0,1. Rows 4..5 for n=2.
    U_323 = np.vstack([U_223, U_123])
    
    # V: 6x11 + 6x6 -> 6x17
    V_323 = np.hstack([V_223, V_123])
    
    # W: 6x11 + 3x6 -> 9x17?
    # W shape (p*n, r).
    # For 2x2x3, W is (3*2, 11) = (6, 11).
    # For 1x2x3, W is (3*1, 6) = (3, 6).
    # Total W for 3x2x3 is (9, 17).
    # Rows 0..5 for n=0,1. Rows 6..8 for n=2.
    W_323 = np.vstack([W_223, W_123])
    
    # 3. Combine 3x3x3 and 3x2x3 to get 3x5x3
    # U: 9x27 + 6x17 -> 15x44
    # U_333 rows 0..8 (n=0,1,2; m=0,1,2)
    # U_323 rows 0..5 (n=0,1,2; m=0,1) -- Wait, 3x2x3 has m=2.
    # In 3x5x3, m=5.
    # U_333 uses m=0,1,2.
    # U_323 uses m=3,4?
    # No, we split A into 3x3 and 3x2.
    # A_L is cols 0,1,2. A_R is cols 3,4.
    # U_333 corresponds to A_L. Its rows are i*3+j for j=0,1,2.
    # In global 3x5, these are i*5+j for j=0,1,2.
    # U_323 corresponds to A_R. Its rows are i*2+j for j=0,1.
    # In global 3x5, these are i*5+3+j for j=0,1.
    
    U_353 = np.zeros((15, 44))
    # Place U_333
    for i in range(3):
        for j in range(3):
            U_353[i*5 + j, 0:27] = U_333[i*3 + j, :]
            
    # Place U_323
    for i in range(3):
        for j in range(2):
            U_353[i*5 + 3 + j, 27:44] = U_323[i*2 + j, :]
            
    # V: 15x27 + 6x17 -> 15x44
    # V_333 (15x27) corresponds to B cols 0,1,2.
    # V_323 (6x17) corresponds to B cols 0,1,2.
    # Both use the same B.
    # So we just hstack.
    V_353 = np.hstack([V_333, V_323])
    
    # W: 9x27 + 9x17 -> 9x44
    # W_333 (9x27) and W_323 (9x17).
    # Both output to C (3x3).
    # So we hstack.
    W_353 = np.hstack([W_333, W_323])
    
    # 4. Extend 3x5x3 to 3x5x6
    # U: 15x44
    U_356 = U_353
    
    # V: 30x44
    V_356 = np.zeros((30, 44))
    for j in range(5):
        for k in range(3):
            V_356[j*6 + k, :] = V_353[j*3 + k, :]
            V_356[j*6 + 3 + k, :] = V_353[j*3 + k, :]
            
    # W: 18x44
    W_356 = np.zeros((18, 44))
    for k in range(3):
        for i in range(3):
            W_356[k*3 + i, :] = W_353[k*3 + i, :]
            W_356[(k+3)*3 + i, :] = W_353[k*3 + i, :]
            
    return {
        "U": U_356.tolist(),
        "V": V_356.tolist(),
        "W": W_356.tolist()
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    # Strassen decomposition for 2x2x2
    # Rank 7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # 1. M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # 2. M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # 3. M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # 4. M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1
    
    # 5. M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # 6. M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = -1
    
    # 7. M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[2, 6] = -1
    
    return U, V, W
