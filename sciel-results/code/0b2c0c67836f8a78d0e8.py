import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 factors (rank 7)
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

    # Identity 2x2x2 factors (rank 4)
    U_id = np.zeros((2, 2, 4), dtype=float)
    V_id = np.zeros((2, 2, 4), dtype=float)
    W_id = np.zeros((2, 2, 4), dtype=float)
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            U_id[i, j, idx] = 1
            V_id[i, j, idx] = 1
            W_id[j, i, idx] = 1  # W[k,i,r] for 2x2x2: output index is (k,i)
    
    # Build 2x4x4 factors via Kronecker: rank = 7*4 = 28
    # U_244[i, j, R] where i in 0..1, j in 0..3, R in 0..27
    # R = 4*s + r where s in 0..6 (Strassen), r in 0..3 (identity)
    U_244 = np.zeros((2, 4, 28), dtype=float)
    V_244 = np.zeros((4, 4, 28), dtype=float)
    W_244 = np.zeros((4, 2, 28), dtype=float)
    
    for i in range(2):
        for j in range(4):
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(4):
                    R = 4 * s + r
                    U_244[i, j, R] = U2[i, j1, s] * U_id[i, j0, r]
        
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(4):
                    R = 4 * s + r
                    V_244[j, k, R] = V2[j1, k1, s] * V_id[j0, k0, r]
        
    for k in range(4):
        for i in range(2):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(4):
                    R = 4 * s + r
                    W_244[k, i, R] = W2[k1, i1, s] * W_id[k0, i0, r]

    # Now build 4x4x8 from two 2x4x4 blocks
    # Block 0: rows 0-1 of A, all of B (4x8), output rows 0-1
    # Block 1: rows 2-3 of A, all of B (4x8), output rows 2-3
    # But B is 4x8, so for each 2x4x4 we need B to be 4x4
    # Actually, 4x4x8: A is 4x4, B is 4x8, C is 4x8
    # Split A into two 2x4 blocks, B into two 4x4 blocks, C into two 2x4 blocks
    # Block 0: A[0:2,:], B[:,0:4], C[0:2,0:4] -> 2x4x4
    # Block 1: A[0:2,:], B[:,4:8], C[0:2,4:8] -> 2x4x4
    # Block 2: A[2:4,:], B[:,0:4], C[2:4,0:4] -> 2x4x4
    # Block 3: A[2:4,:], B[:,4:8], C[2:4,4:8] -> 2x4x4
    
    # Total rank = 4 * 28 = 112? That's worse than 98!
    
    # Alternative: treat as 4x4x8 directly
    # Use 4x4x4 Strassen (rank 49) for each half of B
    # That gives 98, which is what we had.
    
    # Let me try: 4x4x8 = (4x4x4) x 2 in some sense
    # Actually, let's use the 49-rank 4x4x4 and apply it to both halves
    # But can we share? If the two B halves are independent, no sharing possible.
    
    # So 98 is likely optimal for this approach. Let's keep it.
    
    # Build 4x4x4 factors (Rank 49)
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
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
    
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, R] = U4[i, j, R]
                U[i*4 + j, 49 + R] = U4[i, j, R]
                
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j*8 + k, R] = V4[j, k, R]
                V[j*8 + (4+k), 49 + R] = V4[j, k, R]
                
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k*4 + i, R] = W4[k, i, R]
                W[(4+k)*4 + i, 49 + R] = W4[k, i, R]

    # Verification
    if time.time() < deadline - 5.0:
        U3 = U.reshape(4, 4, 98)
        V3 = V.reshape(4, 8, 98)
        W3 = W.reshape(8, 4, 98)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(result, 1.0, atol=1e-10):
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
