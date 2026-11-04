"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 6

    # We use the Kronecker product of:
    # 1. Strassen 2x2x2 decomposition (rank 7)
    # 2. Identity 1x2x3 decomposition (rank 6)
    # Total rank = 7 * 6 = 42
    
    # Strassen 2x2x2 terms
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # Indices: A(2x2) -> 4, B(2x2) -> 4, C(2x2) -> 4
    U_s = [
        [1, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 0, -1, 1, 0, 1, -1],
        [1, 1, 1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 1, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, -1, 1, 0, 1, 1, 1]
    ]
    # Note: Standard Strassen definitions vary by index ordering.
    # We must verify the decomposition.

    # Identity 1x2x3 terms
    # U_id: 2x6 (A is 1x2, so 2 entries)
    # V_id: 6x6 (B is 2x3, so 6 entries)
    # W_id: 6x6 (C is 1x3, so 3 entries? No, C is 1x3. 
    # Wait, 1x2x3 means A(1x2), B(2x3), C(1x3).
    # U has size 1*2 = 2.
    # V has size 2*3 = 6.
    # W has size 3*1 = 3.
    # Rank is 6.
    # Term r: U[i,r], V[j,k,r], W[k,i,r].
    # For identity:
    # r corresponds to (j,k) pair in B?
    # C[k] = sum_j A[j] B[j,k].
    # Term for (j,k): A[j] * B[j,k] * 1.
    # So U[j] = 1, V[j,k] = 1, W[k] = 1.
    
    U_id = np.zeros((2, 6))
    V_id = np.zeros((6, 6))
    W_id = np.zeros((3, 6))
    
    for j in range(2):
        for k in range(3):
            r = j * 3 + k
            U_id[j, r] = 1
            V_id[j * 3 + k, r] = 1
            W_id[k, r] = 1

    # Kronecker Product
    # n=2, m=4, p=6
    # n = 2*1, m = 2*2, p = 2*3
    # A is 2x4. Block structure: 2x2 blocks of 1x2?
    # A[i, j] where i in 0..1, j in 0..3.
    # Let i = i1 * 1 + i2? No.
    # Let i = i1 (0..1).
    # Let j = j1 * 2 + j2 (j1 in 0..1, j2 in 0..1).
    # Let k = k1 * 3 + k2 (k1 in 0..1, k2 in 0..2).
    
    # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Kronecker term:
    # U[i, j, r] = U_s[i, j1, r_s] * U_id[j2, r_id]
    # V[j, k, r] = V_s[j1, k1, r_s] * V_id[j2, k2, r_id]
    # W[k, i, r] = W_s[k1, i, r_s] * W_id[k2, r_id]
    
    # Rank = 7 * 6 = 42.
    
    U = np.zeros((n * m, 42))
    V = np.zeros((m * p, 42))
    W = np.zeros((p * n, 42))
    
    for r_s in range(7):
        for r_id in range(6):
            r = r_s * 6 + r_id
            
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        j = j1 * 2 + j2
                        U[i * 4 + j, r] = U_s[i, j1][r_s] * U_id[j2, r_id]
                        
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    for k1 in range(2):
                        for k2 in range(3):
                            k = k1 * 3 + k2
                            V[j * 6 + k, r] = V_s[j1, k1][r_s] * V_id[j2, k2][r_id]
                            
            for k1 in range(2):
                for k2 in range(3):
                    k = k1 * 3 + k2
                    for i in range(2):
                        W[k * 2 + i, r] = W_s[k1, i][r_s] * W_id[k2, r_id]

    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p, 42):
        # Fallback to identity if Strassen fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
