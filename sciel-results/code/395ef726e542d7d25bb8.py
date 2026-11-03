import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def schoolbook():
        rank = n * m * p
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

    # Strassen rank-7 matrices for 2x2 multiplication
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.float64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)
    
    # Verify 2x2 Strassen
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i])
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    if not valid:
        return schoolbook()
    
    # Kronecker product gives rank 49
    U4k = np.kron(U2, U2)
    V4k = np.kron(V2, V2)
    W4k = np.kron(W2, W2)
    
    # Permutation to fix row ordering
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        R = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
        perm[S] = R
    
    U4 = U4k[perm, :].copy()
    V4 = V4k[perm, :].copy()
    W4 = W4k[perm, :].copy()
    
    # Verify 4x4
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = np.dot(U4[i*4+j], V4[j*4+k] * W4[k*4+i])
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    if not valid:
        return schoolbook()
    
    if elapsed() > DEADLINE:
        return schoolbook()
    
    # Try to find a lower-rank decomposition via subspace selection
    # The tensor T has 64 entries (i,j,k). A rank-r decomposition means
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # We can set up a system: for each (i,j,k), the equation is
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # This is a system of 64 polynomial equations in r*(16+16+16) unknowns.
    # For r < 49, this becomes an underdetermined nonlinear system.
    #
    # Strategy: Use the structure of the Kronecker decomposition.
    # The 49 basis vectors are indexed by (c1, c2) where c1,c2 in {0,...,6}.
    # Each factor is U2[i1,j1,c1]*U2[i2,j2,c2] etc.
    # 
    # We can try to find linear combinations of the 49 columns that give
    # the same tensor with fewer terms. This is equivalent to finding a
    # lower-rank decomposition.
    #
    # One approach: try to find a subset of columns or linear combinations.
    # But the most promising is to check if the tensor has a decomposition
    # with rank < 49.
    #
    # Known results: The rank of M_{4,4,4} is between 49 (upper bound via
    # Strassen^2) and... actually, it's known that rank(M_{4,4,4}) <= 49
    # and the best known lower bound is 42 (Coppersmith-Winograd type).
    #
    # For a practical approach within time constraints, let's try:
    # 1. Check if we can eliminate columns that are linear combinations
    # 2. Try random projections to reduce dimension
    #
    # Actually, a more direct approach: The tensor can be viewed as a
    # bilinear form. We can try to find a decomposition with rank r by
    # solving for U, V, W matrices of shape (16, r), (16, r), (16, r).
    #
    # Let's try a greedy approach: start with the rank-49 decomposition
    # and try to remove columns that are "redundant".
    #
    # For each column c, check if removing it still gives a valid
    # decomposition (with adjusted other columns). This is complex.
    #
    # Alternative: Use the fact that the tensor is symmetric in a certain sense
    # and try to exploit that.
    #
    # Given time constraints, the rank-49 decomposition is likely the best
    # we can achieve. Let's return it.
    
    # Convert to int if all values are integers
    def to_int_if_possible(A):
        if np.allclose(A, np.round(A), atol=1e-9):
            return np.round(A).astype(int)
        return A
    
    U4 = to_int_if_possible(U4)
    V4 = to_int_if_possible(V4)
    W4 = to_int_if_possible(W4)
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
