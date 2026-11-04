import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    This function constructs a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    The decomposition is derived from the Kronecker product of Strassen's rank-7 algorithm for 2x2 matrices.
    The score is based on the rank (lower is better). Rank 49 gives a score of 64/49 ~= 1.306.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Strassen's 2x2x2 algorithm matrices
    # Dimensions: 4x7
    # Indexing convention:
    # U2[row, rank] where row = i*2 + j
    # V2[row, rank] where row = j*2 + k
    # W2[row, rank] where row = k*2 + i (trace convention for output C[i,k])
    
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
    
    # Perform Kronecker product to get 16x49 matrices
    # np.kron(U2, U2) results in rows indexed by (r1, c1, r2, c2) -> r1*4 + r2, etc.
    # We need to map these to the standard row-major indices for 4x4 blocks.
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutation mapping
    # The Kronecker product of two 4x7 matrices results in a 16x49 matrix.
    # The row index in the Kronecker product `R` is `r_outer * 4 + r_inner`.
    # Here `r_outer` corresponds to the row index in the first factor (0-3) and `r_inner` in the second (0-3).
    # However, the logical structure of the 4x4 matrix A is composed of 2x2 blocks.
    # Let the 4x4 matrix A have indices i, j in [0,3].
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # The row index in the task convention is S = i*4 + j = (2*i1 + i2)*4 + (2*j1 + j2) = 8*i1 + 2*j1 + 4*i2 + j2.
    # 
    # In the Kronecker product U4_kron = kron(U2, U2):
    # U2 has rows indexed by (a, b) where a, b in [0,1] representing i1, j1 for the first factor?
    # Actually, let's look at the structure.
    # U2 row r corresponds to input pair (u, v) where u=r//2, v=r%2.
    # kron(U2, U2) row R corresponds to pairs (u1, v1) and (u2, v2).
    # R = r1 * 4 + r2, where r1 = u1*2 + v1, r2 = u2*2 + v2.
    # R = (u1*2 + v1)*4 + (u2*2 + v2) = 8*u1 + 4*v1 + 2*u2 + v2.
    #
    # We want to map the logical indices of the 4x4 matrix:
    # i = 2*u1 + u2
    # j = 2*v1 + v2
    # S = i*4 + j = (2*u1 + u2)*4 + (2*v1 + v2) = 8*u1 + 4*u2 + 2*v1 + v2.
    #
    # So we need a permutation perm such that U4[perm[S], :] = U4_kron[R, :]
    # where R = 8*u1 + 4*v1 + 2*u2 + v2 and S = 8*u1 + 4*u2 + 2*v1 + v2.
    
    perm = np.zeros(16, dtype=int)
    for u1 in range(2):
        for v1 in range(2):
            for u2 in range(2):
                for v2 in range(2):
                    # Calculate R (Kronecker index)
                    r1 = u1 * 2 + v1
                    r2 = u2 * 2 + v2
                    R = r1 * 4 + r2
                    
                    # Calculate S (Task index)
                    i = u1 * 2 + u2
                    j = v1 * 2 + v2
                    S = i * 4 + j
                    
                    perm[S] = R

    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]
    
    # Verification
    # Check if U[i,j] . V[j,k] * W[k,i] == delta_{i,j,k} for all i,j,k
    # Specifically, sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] == 1 if this is the correct term?
    # The tensor T_{ijk} = delta_{i, row(C)}? No.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
    # Decomposition: C[i,k] = sum_r U[i,j, r] ... wait.
    # The factors are U (n*m x rank), V (m*p x rank), W (p*n x rank).
    # The condition is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This is because the tensor of matrix multiplication has entries 1 at (i,j,k) corresponding to A[i,j]B[j,k]C[k,i]?
    # Actually, the standard definition is that the decomposition represents the identity map on the tensor product.
    # The tensor T has T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k] to contribute to C[i,k].
    # The trace convention in the prompt: W[k*n+i, r].
    # So the check is:
    # For all i in 0..3, j in 0..3, k in 0..3:
    #   sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] == 1
    
    valid = True
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_vec = U4[i*4+j]
                    v_vec = V4[j*4+k]
                    w_vec = W4[k*4+i]
                    # Compute dot product
                    val = np.dot(u_vec, v_vec * w_vec)
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

    if not valid:
        # Fallback to schoolbook (Rank 64) - this should not happen
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
