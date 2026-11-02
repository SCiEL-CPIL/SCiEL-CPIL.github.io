import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of M_{n,m,p}."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return (U, V, W) for Strassen decomposition of M_{2,2,2}.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    For n=m=p=2: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A is 2x2: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B is 2x2: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    
    # U rows: A11=0, A12=1, A21=2, A22=3
    # V rows: B11=0, B12=1, B21=2, B22=3
    # W rows: C11=0, C12=1, C21=2, C22=3
    
    # M1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    # V: B11=1, B22=1
    V[0, 0] = 1.0; V[3, 0] = 1.0
    # W: C11=1, C22=1
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2: (A21+A22)B11
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0  # C21
    
    # M3: A11(B12-B22)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0  # C12=1, C22=1
    
    # M4: A22(B21-B11)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0  # C11=1, C21=1
    
    # M5: (A11+A12)B22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0  # C11=-1, C12=1
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0  # C22=1
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0  # C11=1
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions. If (U1,V1,W1) computes C1 = A*B for subproblem 1,
    and (U2,V2,W2) computes C2 = A*B for subproblem 2, then the combined
    decomposition has U = [U1, U2], V = [V1, V2], W = [W1, W2] horizontally.
    """
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix 
    by a 4 by 7 matrix.
    
    The output factor follows the official trace convention W[k*n+i,r].
    
    Returns:
        dict: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Decompose M_{3,4,7} by splitting the first dimension:
    # M_{3,4,7} = M_{2,4,7} + M_{1,4,7}
    
    # M_{1,4,7}: schoolbook, rank = 1*4*7 = 28
    U_147, V_147, W_147 = _schoolbook(1, 4, 7)
    
    # M_{2,4,7}: split second dimension into 2+2
    # M_{2,4,7} = M_{2,2,7} + M_{2,2,7}
    
    # M_{2,2,7}: split third dimension into 3+3+1
    # M_{2,2,7} = M_{2,2,3} + M_{2,2,3} + M_{2,2,1}
    
    # M_{2,2,3}: split third dimension into 2+1
    # M_{2,2,3} = M_{2,2,2} + M_{2,2,1}
    
    # M_{2,2,2}: Strassen, rank 7
    # M_{2,2,1}: schoolbook, rank = 2*2*1 = 4
    
    # Let's build M_{2,2,3}:
    U_222, V_222, W_222 = _strassen_222()
    U_221, V_221, W_221 = _schoolbook(2, 2, 1)
    
    # M_{2,2,3} = M_{2,2,2} + M_{2,2,1}
    # But we need to be careful with indexing. The W matrix for M_{2,2,3} has shape (3*2, rank) = (6, rank)
    # M_{2,2,2} W has shape (2*2, 7) = (4, 7)
    # M_{2,2,1} W has shape (1*2, 4) = (2, 4)
    # We need to embed these into the correct positions.
    
    # For M_{2,2,3}, the output indices are (k, i) where k in [0,3), i in [0,2)
    # W index = k*n + i = k*2 + i
    # M_{2,2,2} covers k in [0,2), so W indices 0,1,2,3
    # M_{2,2,1} covers k=2, so W indices 4,5
    
    # Let's create the full W for M_{2,2,3}
    rank_223 = 7 + 4  # 11
    U_223 = np.zeros((2*2, rank_223))
    V_223 = np.zeros((2*3, rank_223))
    W_223 = np.zeros((3*2, rank_223))
    
    # Embed M_{2,2,2}
    U_223[:, :7] = U_222
    V_223[:, :7] = V_222  # V_222 has shape (4, 7), V_223 has shape (6, 11)
    # V_222 indices: j*p+k where p=2, so j in [0,2), k in [0,2)
    # V_223 indices: j*p+k where p=3, so j in [0,2), k in [0,3)
    # M_{2,2,2} uses k in [0,2), so V_222 row j*2+k maps to V_223 row j*3+k
    # We need to remap the V indices
    V_223_remapped = np.zeros((6, 7))
    for j in range(2):
        for k in range(2):
            V_223_remapped[j*3+k, :] = V_222[j*2+k, :]
    V_223[:, :7] = V_223_remapped
    
    # W_222 has shape (4, 7), indices k*2+i for k in [0,2), i in [0,2)
    # W_223 has shape (6, 11), indices k*2+i for k in [0,3), i in [0,2)
    # M_{2,2,2} W covers k in [0,2), so rows 0,1,2,3
    W_223[:4, :7] = W_222
    
    # Embed M_{2,2,1}
    # U_221 has shape (4, 4), same as U_223 first 4 rows
    U_223[:, 7:] = U_221
    # V_221 has shape (2, 4), indices j*1+k = j for j in [0,2), k=0
    # V_223 has shape (6, 11), indices j*3+k
    # M_{2,2,1} uses k=2 (the third column), so V_221 row j maps to V_223 row j*3+2
    V_223_remapped2 = np.zeros((6, 4))
    for j in range(2):
        V_223_remapped2[j*3+2, :] = V_221[j, :]
    V_223[:, 7:] = V_223_remapped2
    
    # W_221 has shape (2, 4), indices k*2+i for k=0, i in [0,2)
    # But in M_{2,2,3}, k=2 for this subproblem, so W index = 2*2+i = 4+i
    W_223[4:, 7:] = W_221
    
    # Now M_{2,2,7} = M_{2,2,3} + M_{2,2,3} + M_{2,2,1}
    # Split p=7 into 3+3+1
    # M_{2,2,3} #1 covers k in [0,3)
    # M_{2,2,3} #2 covers k in [3,6)
    # M_{2,2,1} covers k=6
    
    rank_227 = 11 + 11 + 4  # 26
    U_227 = np.zeros((4, rank_227))
    V_227 = np.zeros((2*7, rank_227))
    W_227 = np.zeros((7*2, rank_227))
    
    # Embed first M_{2,2,3}
    U_227[:, :11] = U_223
    # V_223 has shape (6, 11), indices j*3+k for j in [0,2), k in [0,3)
    # V_227 has shape (14, 26), indices j*7+k for j in [0,2), k in [0,7)
    # First M_{2,2,3} covers k in [0,3)
    for j in range(2):
        for k in range(3):
            V_227[j*7+k, :11] = V_223[j*3+k, :]
    # W_223 has shape (6, 11), indices k*2+i for k in [0,3), i in [0,2)
    # W_227 has shape (14, 26), indices k*2+i for k in [0,7), i in [0,2)
    W_227[:6, :11] = W_223
    
    # Embed second M_{2,2,3}
    U_227[:, 11:22] = U_223
    for j in range(2):
        for k in range(3):
            V_227[j*7+(3+k), 11:22] = V_223[j*3+k, :]
    for k in range(3):
        for i in range(2):
            W_227[(3+k)*2+i, 11:22] = W_223[k*2+i, :]
    
    # Embed M_{2,2,1}
    U_227[:, 22:] = U_221
    for j in range(2):
        V_227[j*7+6, 22:] = V_221[j, :]
    for i in range(2):
        W_227[6*2+i, 22:] = W_221[i, :]
    
    # Now M_{2,4,7} = M_{2,2,7} + M_{2,2,7}
    # Split m=4 into 2+2
    rank_247 = 26 + 26  # 52
    U_247 = np.zeros((2*4, rank_247))
    V_247 = np.zeros((4*7, rank_247))
    W_247 = np.zeros((7*2, rank_247))
    
    # Embed first M_{2,2,7} (covers j in [0,2))
    U_247[:4, :26] = U_227
    # V_227 has shape (14, 26), indices j*7+k for j in [0,2), k in [0,7)
    # V_247 has shape (28, 52), indices j*7+k for j in [0,4), k in [0,7)
    # First M_{2,2,7} covers j in [0,2)
    for j in range(2):
        for k in range(7):
            V_247[j*7+k, :26] = V_227[j*7+k, :]
    W_247[:, :26] = W_227
    
    # Embed second M_{2,2,7} (covers j in [2,4))
    U_247[4:, 26:] = U_227
    for j in range(2):
        for k in range(7):
            V_247[(2+j)*7+k, 26:] = V_227[j*7+k, :]
    W_247[:, 26:] = W_227
    
    # Now M_{3,4,7} = M_{2,4,7} + M_{1,4,7}
    # Split n=3 into 2+1
    rank_347 = 52 + 28  # 80
    U_347 = np.zeros((3*4, rank_347))
    V_347 = np.zeros((4*7, rank_347))
    W_347 = np.zeros((7*3, rank_347))
    
    # Embed M_{2,4,7} (covers i in [0,2))
    U_347[:8, :52] = U_247
    V_347[:, :52] = V_247
    # W_247 has shape (14, 52), indices k*2+i for k in [0,7), i in [0,2)
    # W_347 has shape (21, 80), indices k*3+i for k in [0,7), i in [0,3)
    # M_{2,4,7} covers i in [0,2)
    for k in range(7):
        for i in range(2):
            W_347[k*3+i, :52] = W_247[k*2+i, :]
    
    # Embed M_{1,4,7} (covers i=2)
    U_347[8:, 52:] = U_147
    V_347[:, 52:] = V_147
    # W_147 has shape (7, 28), indices k*1+i = k for k in [0,7), i=0
    # W_347 has shape (21, 80), indices k*3+i
    # M_{1,4,7} covers i=2, so W index = k*3+2
    for k in range(7):
        W_347[k*3+2, 52:] = W_147[k, :]
    
    return {
        "U": U_347.tolist(),
        "V": V_347.tolist(),
        "W": W_347.tolist()
    }
