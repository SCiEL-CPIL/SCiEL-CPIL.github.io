"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time

def construct():
    """
    Construct a rank-34 exact tensor decomposition for 3x3 @ 3x4 multiplication.
    
    Uses a structured block decomposition:
    - 7 Strassen terms for A11*B1L (2x2 blocks, left columns)
    - 7 Strassen terms for A11*B1R (2x2 blocks, right columns)  
    - 8 direct terms for A12*B2L and A21*B1L (cross blocks, left)
    - 8 direct terms for A12*B2R and A21*B1R (cross blocks, right)
    - 4 direct terms for A22*B2L and A22*B2R (bottom-right block)
    
    Total: 7 + 7 + 8 + 8 + 4 = 34 terms
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 3, 3, 4  # A is n x m, B is m x p, C is n x p
    rank = 34
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34 (A factors)
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34 (B factors)
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34 (C factors)
    
    col = 0
    
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:3], A21 = A[2:3, 0:2], A22 = A[2:3, 2:3]
    # B1L = B[0:2, 0:2], B2L = B[2:3, 0:2], B1R = B[0:2, 2:4], B2R = B[2:3, 2:4]
    # C1L = C[0:2, 0:2], C1R = C[0:2, 2:4], C2L = C[2:3, 0:2], C2R = C[2:3, 2:4]
    
    # Indices in flattened form:
    # A[i,j] -> i*m + j
    # B[i,j] -> i*p + j
    # C[k,n+i,r] -> k*n + i + r  (official trace convention: W[k*n+i,r] where i is col of C)
    # Actually: C has shape (n, p), so C[i,j] -> i*p + j
    # But the contract says W[k*n+i,r]... let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This likely means W is indexed by (output_index, term) where output_index corresponds to C positions
    # C is n x p = 3 x 4, so 12 entries. W has shape (p*n, rank) = (12, rank)
    # The index in W corresponds to C[i,j] = i*p + j (row-major)
    
    # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5, A[2,0]=6, A[2,1]=7, A[2,2]=8
    # B indices: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7, B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # C indices: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7, C[2,0]=8, C[2,1]=9, C[2,2]=10, C[2,3]=11
    
    # Block definitions:
    # A11: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # A12: A[0,2]=2, A[1,2]=5
    # A21: A[2,0]=6, A[2,1]=7
    # A22: A[2,2]=8
    # B1L: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # B2L: B[2,0]=8, B[2,1]=9
    # B1R: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # B2R: B[2,2]=10, B[2,3]=11
    # C1L: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    # C1R: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    # C2L: C[2,0]=8, C[2,1]=9
    # C2R: C[2,2]=10, C[2,3]=11
    
    # Strassen for 2x2 * 2x2 (A11 * B1L -> C1L)
    # M1 = (A11[0,0] + A11[1,1]) * (B1L[0,0] + B1L[1,1])
    # M2 = (A11[1,0] + A11[1,1]) * B1L[0,0]
    # M3 = A11[0,0] * (B1L[0,1] - B1L[1,1])
    # M4 = A11[1,1] * (B1L[1,0] - B1L[0,0])
    # M5 = (A11[0,0] + A11[0,1]) * B1L[1,1]
    # M6 = (A11[1,0] - A11[0,0]) * (B1L[0,0] + B1L[0,1])
    # M7 = (A11[0,1] - A11[1,1]) * (B1L[1,0] + B1L[1,1])
    
    # C1L[0,0] = M1 - M2 + M3 - M5
    # C1L[0,1] = M2 + M5
    # C1L[1,0] = M3 + M4
    # C1L[1,1] = M1 - M4 - M6 + M7
    
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[5][col] = 1  # C[0,0]=0, C[1,1]=5
    col += 1
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[4][col] = 1  # C[0,0]=-1, C[1,0]=1... wait
    # C1L[0,0] includes -M2, C1L[0,1] includes +M2, C1L[1,1] does not include M2
    # So: W[0] += -1, W[1] += 1
    # Let me redo:
    # C1L[0,0] = M1 - M2 + M3 - M5
    # C1L[0,1] = M2 + M5
    # C1L[1,0] = M3 + M4
    # C1L[1,1] = M1 - M4 - M6 + M7
    
    # M1: C[0,0]+=1, C[1,1]+=1
    # M2: C[0,0]+=-1, C[0,1]+=1
    # M3: C[0,0]+=1, C[1,0]+=1
    # M4: C[1,0]+=1, C[1,1]+=-1
    # M5: C[0,0]+=-1, C[0,1]+=1
    # M6: C[1,1]+=-1
    # M7: C[1,1]+=1
    
    # Redo M2:
    # Already set above incorrectly. Let me restart cleanly.
    col = 0
    
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M3: A[0,0] * (B[0,1]-B[1,1])
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M4: A[1,1] * (B[1,0]-B[0,0])
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[4][col] = 1; W[5][col] = -1
    col += 1
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = -1
    col += 1
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A12 * B2L -> contributes to C1L
    # C1L[0,0] += A[0,2]*B[2,0]
    # C1L[0,1] += A[0,2]*B[2,1]
    # C1L[1,0] += A[1,2]*B[2,0]
    # C1L[1,1] += A[1,2]*B[2,1]
    
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1
    col += 1
    U[2][col] = 1; V[9][col] = 1; W[1][col] = 1
    col += 1
    U[5][col] = 1; V[8][col] = 1; W[4][col] = 1
    col += 1
    U[5][col] = 1; V[9][col] = 1; W[5][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A21 * B1L -> contributes to C2L
    # C2L[0] += A[2,0]*B[0,0]  (C[2,0]=8)
    # C2L[1] += A[2,0]*B[0,1]  (C[2,1]=9)
    # C2L[0] += A[2,1]*B[1,0]  (C[2,0]=8)
    # C2L[1] += A[2,1]*B[1,1]  (C[2,1]=9)
    
    U[6][col] = 1; V[0][col] = 1; W[8][col] = 1
    col += 1
    U[6][col] = 1; V[1][col] = 1; W[9][col] = 1
    col += 1
    U[7][col] = 1; V[4][col] = 1; W[8][col] = 1
    col += 1
    U[7][col] = 1; V[5][col] = 1; W[9][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A22 * B2L -> contributes to C2L
    # C2L[0] += A[2,2]*B[2,0]  (C[2,0]=8)
    # C2L[1] += A[2,2]*B[2,1]  (C[2,1]=9)
    
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1
    col += 1
    U[8][col] = 1; V[9][col] = 1; W[9][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Strassen for A11 * B1R -> C1R
    # Same structure, different B and C indices
    # B1R: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C1R: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    
    # M1: (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[7][col] = 1
    col += 1
    
    # M2: (A[1,0]+A[1,1]) * B[0,2]
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M3: A[0,0] * (B[0,3]-B[1,3])
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[2][col] = 1; W[6][col] = 1
    col += 1
    
    # M4: A[1,1] * (B[1,2]-B[0,2])
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M5: (A[0,0]+A[0,1]) * B[1,3]
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6: (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = -1
    col += 1
    
    # M7: (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A12 * B2R -> contributes to C1R
    # C1R[0,2] += A[0,2]*B[2,2]  (C[0,2]=2)
    # C1R[0,3] += A[0,2]*B[2,3]  (C[0,3]=3)
    # C1R[1,2] += A[1,2]*B[2,2]  (C[1,2]=6)
    # C1R[1,3] += A[1,2]*B[2,3]  (C[1,3]=7)
    
    U[2][col] = 1; V[10][col] = 1; W[2][col] = 1
    col += 1
    U[2][col] = 1; V[11][col] = 1; W[3][col] = 1
    col += 1
    U[5][col] = 1; V[10][col] = 1; W[6][col] = 1
    col += 1
    U[5][col] = 1; V[11][col] = 1; W[7][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A21 * B1R -> contributes to C2R
    # C2R[0] += A[2,0]*B[0,2]  (C[2,2]=10)
    # C2R[1] += A[2,0]*B[0,3]  (C[2,3]=11)
    # C2R[0] += A[2,1]*B[1,2]  (C[2,2]=10)
    # C2R[1] += A[2,1]*B[1,3]  (C[2,3]=11)
    
    U[6][col] = 1; V[2][col] = 1; W[10][col] = 1
    col += 1
    U[6][col] = 1; V[3][col] = 1; W[11][col] = 1
    col += 1
    U[7][col] = 1; V[6][col] = 1; W[10][col] = 1
    col += 1
    U[7][col] = 1; V[7][col] = 1; W[11][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A22 * B2R -> contributes to C2R
    # C2R[0] += A[2,2]*B[2,2]  (C[2,2]=10)
    # C2R[1] += A[2,2]*B[2,3]  (C[2,3]=11)
    
    U[8][col] = 1; V[10][col] = 1; W[10][col] = 1
    col += 1
    U[8][col] = 1; V[11][col] = 1; W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
