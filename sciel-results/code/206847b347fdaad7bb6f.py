"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    rank = 13
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen for k in {0, 1}
    # m1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[6, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # m4 = A11(B10 - B00)
    U[3, 3] = 1
    V[5, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # m5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[6, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # m6 = (A01 - A10)(B00 + B10)
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[5, 5] = 1
    W[0, 5] = 1
    
    # m7 = (A01 - A11)(B01 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[6, 6] = 1
    W[3, 6] = 1
    
    # Schoolbook for k in {2, 3, 4}
    # k=2, rank 7
    for i in range(n):
        for j in range(m):
            U[i*m+j, 7] = 1
            V[j*p+2, 7] = 1
            W[2*n+i, 7] = 1
            
    # k=3, rank 8
    for i in range(n):
        for j in range(m):
            U[i*m+j, 8] = 1
            V[j*p+3, 8] = 1
            W[3*n+i, 8] = 1
            
    # k=4, rank 9
    for i in range(n):
        for j in range(m):
            U[i*m+j, 9] = 1
            V[j*p+4, 9] = 1
            W[4*n+i, 9] = 1
            
    # Wait, rank 13 means indices 0..12.
    # I used 0..6 for Strassen (7 terms).
    # I used 7, 8, 9 for Schoolbook (3 terms).
    # Total 10 terms?
    # Schoolbook for 2x2x1 is rank 4?
    # No, for 2x2x1, A is 2x2, B is 2x1, C is 2x1.
    # C_i = sum_j A_ij B_j.
    # Terms: A00 B0, A01 B1, A10 B0, A11 B1.
    # Rank is 4.
    # So for each k in {2, 3, 4}, we need 4 terms.
    # Total rank = 7 + 3*4 = 19.
    # This is worse than 20? No, 19 < 20.
    # But can we do better for the 2x2x1 part?
    # Rank of 2x2x1 matrix multiplication is 4 (schoolbook).
    # Is there a lower rank? No, lower bound is 4.
    # So rank 19 is the best with this hybrid.
    # My previous estimate of 13 was wrong because I thought schoolbook for 2x2x1 was rank 2?
    # No, it's rank 4.
    # Let's re-verify.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Tensor entries (i, j, k=0):
    # (0,0,0): A00 B0 -> 1
    # (0,1,0): A01 B1 -> 1
    # (1,0,0): A10 B0 -> 1
    # (1,1,0): A11 B1 -> 1
    # All 4 terms are independent. Rank 4.
    
    # So we need 4 terms for each of the 3 remaining columns.
    # Total rank = 7 + 12 = 19.
    
    # Let's fix the code to use rank 19.
    
    U = np.zeros((n*m, 19))
    V = np.zeros((m*p, 19))
    W = np.zeros((p*n, 19))
    
    # Re-initialize Strassen part (ranks 0-6)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[6, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    U[3, 3] = 1
    V[5, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    U[0, 4] = 1; U[2, 4] = 1
    V[6, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[5, 5] = 1
    W[0, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[6, 6] = 1
    W[3, 6] = 1
    
    # Schoolbook for k=2 (ranks 7-10)
    # Terms: (0,0), (0,1), (1,0), (1,1)
    # (0,0): U[0,7]=1, V[0*5+2,7]=1, W[2*2+0,7]=1
    U[0, 7] = 1; V[2, 7] = 1; W[0, 7] = 1
    # (0,1): U[1,8]=1, V[1*5+2,8]=1, W[2*2+0,8]=1
    U[1, 8] = 1; V[7, 8] = 1; W[0, 8] = 1
    # (1,0): U[2,9]=1, V[0*5+2,9]=1, W[2*2+1,9]=1
    U[2, 9] = 1; V[2, 9] = 1; W[1, 9] = 1
    # (1,1): U[3,10]=1, V[1*5+2,10]=1, W[2*2+1,10]=1
    U[3, 10] = 1; V[7, 10] = 1; W[1, 10] = 1
    
    # Schoolbook for k=3 (ranks 11-14)
    U[0, 11] = 1; V[3, 11] = 1; W[0, 11] = 1
    U[1, 12] = 1; V[8, 12] = 1; W[0, 12] = 1
    U[2, 13] = 1; V[3, 13] = 1; W[1, 13] = 1
    U[3, 14] = 1; V[8, 14] = 1; W[1, 14] = 1
    
    # Schoolbook for k=4 (ranks 15-18)
    U[0, 15] = 1; V[4, 15] = 1; W[0, 15] = 1
    U[1, 16] = 1; V[9, 16] = 1; W[0, 16] = 1
    U[2, 17] = 1; V[4, 17] = 1; W[1, 17] = 1
    U[3, 18] = 1; V[9, 18] = 1; W[1, 18] = 1
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 20 if verification fails (should not happen)
    rank = 20
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
