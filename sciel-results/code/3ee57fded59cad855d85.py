import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    Uses the rank-7 Strassen decomposition for 2x2x2 and takes the Kronecker product
    to obtain a rank-49 decomposition for 4x4x4. Careful row permutations are applied
    to ensure the linear indexing matches the task's convention:
    U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    
    The score is based on the rank (lower is better). Rank 49 gives a score of 64/49 ~= 1.306.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Strassen's 2x2x2 algorithm matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # U2[i*2+j, r] = coefficient of A[i,j] in term r
    # V2[j*2+k, r] = coefficient of B[j,k] in term r
    # W2[k*2+i, r] = coefficient of C[i,k] in term r (trace convention)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker products
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Permutation to map Kronecker row indices to task's linear indices
    # 
    # In np.kron(U2, U2), row index R = I*4 + i where:
    #   I = r_out*2 + c_out (row in outer 2x2 block matrix)
    #   i = r_in*2 + c_in   (row in inner 2x2 block)
    #   So R = (r_out*2 + c_out)*4 + (r_in*2 + c_in)
    #        = r_out*8 + c_out*4 + r_in*2 + c_in
    #
    # Task index S for U[i*4+j]:
    #   i = r_out*2 + r_in
    #   j = c_out*2 + c_in
    #   S = (r_out*2 + r_in)*4 + (c_out*2 + c_in)
    #     = r_out*8 + r_in*4 + c_out*2 + c_in
    #
    # We need perm[S] = R
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        r_out = S // 8
        r_in = (S // 4) % 2
        c_out = (S // 2) % 2
        c_in = S % 2
        
        R = r_out * 8 + c_out * 4 + r_in * 2 + c_in
        perm[S] = R
    
    U4 = U4_raw[perm, :]
    V4 = V4_raw[perm, :]
    W4 = W4_raw[perm, :]
    
    # Verify the decomposition
    valid = True
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = U4[i*4+j]
                    v = V4[j*4+k]
                    w = W4[k*4+i]
                    s = np.dot(u, v * w)
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to schoolbook (Rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
