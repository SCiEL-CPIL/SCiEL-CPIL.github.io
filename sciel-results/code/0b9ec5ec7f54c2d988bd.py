import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U2[i, j, r], V2[j, k, r], W2[k, i, r]
    # r is 0..6
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)
    
    # r=0
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Build 4x4x4 factors (Rank 49)
    # U4[i, j, R] where R = 7*s + r
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
    # Vectorized construction of 4x4x4 from 2x2x2
    # U4[i, j, 7*s + r] = U2[i//2, j//2, s] * U2[i%2, j%2, r]
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(7):
                    U4[i, j, 7*s + r] = U2[i1, j1, s] * U2[i0, j0, r]
                    
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(7):
                    V4[j, k, 7*s + r] = V2[j1, k1, s] * V2[j0, k0, r]
                    
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(7):
                    W4[k, i, 7*s + r] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Fill U (16x98)
    # U[i*4+j, R]
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, R] = U4[i, j, R]
                U[i*4 + j, 49 + R] = U4[i, j, R]
                
    # Fill V (32x98)
    # V[j*8+k, R]
    # First half (k < 4): V[j*8+k, R] = V4[j, k, R]
    # Second half (k >= 4): V[j*8+k, R] = V4[j, k-4, R]
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j*8 + k, R] = V4[j, k, R]
                V[j*8 + (4+k), 49 + R] = V4[j, k, R]
                
    # Fill W (32x98)
    # W[k*4+i, R]
    # First half (k < 4): W[k*4+i, R] = W4[k, i, R]
    # Second half (k >= 4): W[k*4+i, R] = W4[k-4, i, R]
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k*4 + i, R] = W4[k, i, R]
                W[(4+k)*4 + i, 49 + R] = W4[k, i, R]

    # Quick Verification (Sample)
    # Check a few random indices to ensure correctness before returning
    # If time allows, do a full check
    if time.time() < deadline - 5.0:
        U_arr = U
        V_arr = V
        W_arr = W
        # Reshape for einsum
        U3 = U_arr.reshape(4, 4, 98)
        V3 = V_arr.reshape(4, 8, 98)
        W3 = W_arr.reshape(8, 4, 98)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Using einsum
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # Check if all are 1.0
        if not np.allclose(result, 1.0, atol=1e-10):
            # Fallback to Schoolbook if Strassen fails
            rank_school = n * m * p
            U_s = np.zeros((n*m, rank_school))
            V_s = np.zeros((m*p, rank_school))
            W_s = np.zeros((p*n, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m + j, col] = 1
                        V_s[j*p + k, col] = 1
                        W_s[k*n + i, col] = 1
                        col += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
