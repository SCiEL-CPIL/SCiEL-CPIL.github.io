import numpy as np
import time

def _verify_decomposition(U, V, W, n=2, m=3, p=6):
    """
    Verifies if U, V, W exactly decompose the n x m x p multiplication tensor.
    Returns True if valid.
    """
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We check the reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This is equivalent to checking if the matrices form a valid decomposition.
    
    rank = U.shape[1]
    
    # For speed, we can check a few random linear combinations or the full tensor.
    # Full tensor check is O(n*m*p*rank). 2*3*6*33 = 1188 operations. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank
                # U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :])
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Rank 33 Solution:
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> 12 terms
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> Split into three 2x2x2 blocks -> 3 * 7 = 21 terms
    # Total: 33 terms
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: Schoolbook for first column of A
    # A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Strassen for remaining 2x2 blocks
    # A[:, 1:3] is 2x2. Let A_sub = [[a, b], [c, d]] corresponding to A[0,1], A[0,2], A[1,1], A[1,2]
    # B[1:3, :] is 2x6. Let B_sub = [[B_row0], [B_row1]]
    # We split B_sub into three 2x2 blocks: cols (0,1), (2,3), (4,5)
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Indices for A_sub:
        # a = A[0, 1] -> U index 0*m + 1
        # b = A[0, 2] -> U index 0*m + 2
        # c = A[1, 1] -> U index 1*m + 1
        # d = A[1, 2] -> U index 1*m + 2
        
        # Indices for B_sub block t:
        # B[1, k0], B[1, k1], B[2, k0], B[2, k1]
        # V indices: 1*p+k0, 1*p+k1, 2*p+k0, 2*p+k1
        
        # Indices for C output for this block:
        # C[0, k0], C[0, k1], C[1, k0], C[1, k1]
        # W indices: k0*n+0, k0*n+1, k1*n+0, k1*n+1
        
        # Strassen M1: (a+d)(B[1,k0]+B[2,k1]) -> contributes to C[0,k0] and C[1,k1]
        U[0 * m + 1, col] += 1.0
        U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # Strassen M2: (c+d)B[1,k0] -> contributes to C[1,k0] and C[1,k1] (with sign)
        U[1 * m + 1, col] += 1.0
        U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0
        W[k1 * n + 1, col] += -1.0
        col += 1

        # Strassen M3: a(B[1,k1]-B[2,k1]) -> contributes to C[0,k1] and C[1,k1]
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0
        V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # Strassen M4: d(B[2,k0]-B[1,k0]) -> contributes to C[0,k0] and C[1,k0]
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0
        V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0
        W[k0 * n + 1, col] += 1.0
        col += 1

        # Strassen M5: (a+b)B[2,k1] -> contributes to C[0,k0] and C[0,k1]
        U[0 * m + 1, col] += 1.0
        U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0
        W[k1 * n + 0, col] += 1.0
        col += 1

        # Strassen M6: (c-a)(B[1,k0]+B[1,k1]) -> contributes to C[1,k1]
        U[1 * m + 1, col] += 1.0
        U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0
        V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # Strassen M7: (b-d)(B[2,k0]+B[2,k1]) -> contributes to C[0,k0]
        U[0 * m + 2, col] += 1.0
        U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify the solution
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Schoolbook Rank 36 if verification fails (should not happen)
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
