import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: Schoolbook Rank 36
    rank_fallback = n * m * p
    U_fallback = np.zeros((n * m, rank_fallback))
    V_fallback = np.zeros((m * p, rank_fallback))
    W_fallback = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j, col] = 1
                V_fallback[j * p + k, col] = 1
                W_fallback[k * n + i, col] = 1
                col += 1
                
    # Strategy: Split j=0 (Rank 1) and j in {1,2} (Rank 21 via 3x Strassen)
    # Total Rank 22
    
    # Define Strassen vectors for 2x2x2
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00+C11
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # Term 2: (A10+A11)B00 -> C10
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1 # k=1, i=0 -> 1*2+0=2
    
    # Term 3: A00(B10-B11) -> C01
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1 # k=1, i=0? No C01 is k=1, i=0. Wait.
    # C[i,k]. i=0, k=1. W index k*n+i = 1*2+0=2?
    # My W_s index is k*2+i.
    # C01 -> i=0, k=1. Index 2.
    # C10 -> i=1, k=0. Index 0*2+1=1.
    # C11 -> i=1, k=1. Index 1*2+1=3.
    # C00 -> i=0, k=0. Index 0.
    
    # Let's re-verify W indices for Strassen terms.
    # T1: C00+C11. W[0]=1, W[3]=1. Correct.
    # T2: C10. i=1, k=0. W[1]=1. Correct.
    # T3: C01. i=0, k=1. W[2]=1. Correct.
    # T4: A11 B11 -> C11-C00? No, M4 = A11 B11.
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M6
    # So M4 contributes +1 to C01 and -1 to C00?
    # Wait, standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11 B11
    # M5 = (A00+A01)B11
    # M6 = (A01-A10)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    
    # C00 = M1 - M4 + M5
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M6
    
    # So:
    # T1: W[0]=1, W[3]=1
    # T2: W[1]=1
    # T3: W[2]=1
    # T4: W[2]=1, W[0]=-1
    # T5: W[0]=1
    # T6: W[3]=1
    # T7: W[1]=1
    
    # Let's fill W_s correctly.
    W_s = np.zeros((4, 7))
    W_s[0, 0] = 1; W_s[3, 0] = 1
    W_s[1, 1] = 1
    W_s[2, 2] = 1
    W_s[2, 3] = 1; W_s[0, 3] = -1
    W_s[0, 4] = 1
    W_s[3, 5] = 1
    W_s[1, 6] = 1
    
    # U_s and V_s
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    
    # T1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    
    # T2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    
    # T3
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    
    # T4
    U_s[3, 3] = 1
    V_s[3, 3] = 1
    
    # T5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    
    # T6
    U_s[1, 5] = 1; U_s[2, 5] = -1
    V_s[2, 5] = 1; V_s[3, 5] = 1
    
    # T7
    U_s[2, 6] = 1; U_s[0, 6] = -1
    V_s[0, 6] = 1; V_s[1, 6] = 1
    
    # Verify Strassen decomposition
    valid_strassen = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r]
                if abs(s - 1) > 1e-9:
                    valid_strassen = False
    
    if not valid_strassen:
        # Fallback to Rank 25 (1 + 24)
        rank_target = 25
        U = np.zeros((n*m, rank_target))
        V = np.zeros((m*p, rank_target))
        W = np.zeros((p*n, rank_target))
        
        # Term 0: j=0
        U[:, 0] = 1 # i*3+0
        V[0:6, 0] = 1 # 0*6+k
        W[:, 0] = 1 # k*2+i
        
        # Terms 1-24: Schoolbook for j in {1,2}
        col = 1
        for i in range(n):
            for j in [1, 2]:
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank 22 Solution
    rank_target = 22
    U = np.zeros((n*m, rank_target))
    V = np.zeros((m*p, rank_target))
    W = np.zeros((p*n, rank_target))
    
    # Term 0: j=0
    U[:, 0] = 1
    V[0:6, 0] = 1
    W[:, 0] = 1
    
    # Terms 1-21: 3 blocks of Strassen for j in {1,2}
    for l in range(3):
        for c in range(7):
            r = 1 + l*7 + c
            for i in range(2):
                for jp in range(2): # j' in 0,1 -> j in 1,2
                    j = jp + 1
                    for kp in range(2): # k' in 0,1 -> k in 2l, 2l+1
                        k = 2*l + kp
                        
                        # U[i, j]
                        U[i*m + j, r] = U_s[i*2+jp, c]
                        # V[j, k]
                        V[j*p + k, r] = V_s[jp*2+kp, c]
                        # W[k, i]
                        W[k*n + i, r] = W_s[kp*2+i, c]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
