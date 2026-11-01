"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    # Strassen 2x2x2 decomposition (rank 7)
    # For C = A * B where A is 2x2, B is 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A indices: A[i,j] -> I = i*m + j
    #   A00: I=0, A01: I=1, A10: I=2, A11: I=3
    # B indices: B[j,k] -> J = j*p + k
    #   For 2x2: B00: J=0, B01: J=1, B10: J=2, B11: J=3
    # C indices: C[i,k] -> K = k*n + i
    #   For 2x2: C00: K=0, C01: K=1, C10: K=2, C11: K=3

    # Strassen coefficients for 2x2x2
    # Each term r has U[4], V[4], W[4]
    # U: A coefficients, V: B coefficients, W: C coefficients
    strassen_U = np.array([
        [1, 1, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 1, 0, 0, 1],   # A01
        [1, 1, 0, 1, 1, -1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ], dtype=float)
    strassen_V = np.array([
        [1, 1, 1, -1, 0, 1, 1],  # B00
        [1, 0, 1, 0, 1, 0, 1],   # B01
        [1, 0, -1, -1, 0, 1, 0], # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ], dtype=float)
    strassen_W = np.array([
        [1, 0, 0, -1, 1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [-1, -1, 1, 0, 0, 1, 1]  # C11
    ], dtype=float)

    # Verify Strassen 2x2x2
    # T[I,J,K] should be 1 for (I=0,J=0,K=0), (I=0,J=1,K=1), (I=1,J=2,K=0), (I=1,J=3,K=1)
    # Wait: C[i,k] -> K = k*n + i
    # C00: i=0,k=0 -> K=0
    # C01: i=0,k=1 -> K=1
    # C10: i=1,k=0 -> K=2
    # C11: i=1,k=1 -> K=3
    # Valid: (I=0,J=0,K=0), (I=0,J=1,K=1), (I=1,J=2,K=0)? No...
    # T[I,J,K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i
    # (i=0,j=0,k=0): I=0, J=0, K=0
    # (i=0,j=0,k=1): I=0, J=1, K=1
    # (i=0,j=1,k=0): I=1, J=2, K=0
    # (i=0,j=1,k=1): I=1, J=3, K=1
    # (i=1,j=0,k=0): I=2, J=0, K=2
    # (i=1,j=0,k=1): I=2, J=1, K=3
    # (i=1,j=1,k=0): I=3, J=2, K=2
    # (i=1,j=1,k=1): I=3, J=3, K=3

    # For 2x2x2, the valid (I,J,K) triples are:
    # (0,0,0), (0,1,1), (1,2,0), (1,3,1), (2,0,2), (2,1,3), (3,2,2), (3,3,3)

    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    I = i * 2 + j
                    J = j * 2 + k
                    K = k * 2 + i
                    val = 0.0
                    for r in range(7):
                        val += strassen_U[I, r] * strassen_V[J, r] * strassen_W[K, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check invalid positions give 0
        for I in range(4):
            for J in range(4):
                for K in range(4):
                    # Check if this is a valid position
                    valid = False
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                if I == i*2+j and J == j*2+k and K == k*2+i:
                                    valid = True
                    if not valid:
                        val = 0.0
                        for r in range(7):
                            val += strassen_U[I, r] * strassen_V[J, r] * strassen_W[K, r]
                        if abs(val) > 1e-9:
                            return False
        return True

    assert verify_strassen(), "Strassen verification failed"

    # For p=15, we can't directly use Strassen because the K index has stride n=2,
    # and the J index has stride p=15. The Strassen decomposition works for p=2.
    #
    # Strategy: Decompose p=15 into blocks. Since Strassen works for p=2, 
    # we can handle 7 blocks of size 2 (covering k=0..13) and 1 block of size 1.
    #
    # For a block of k values {k0, k0+1}, the relevant J values are j*15+k0 and j*15+k0+1
    # and K values are (k0)*2+i and (k0+1)*2+i.
    #
    # The Strassen decomposition for a 2-column block maps:
    # B block: rows j=0,1, cols k=k0,k0+1
    #   J indices: 0*15+k0, 0*15+k0+1, 1*15+k0, 1*15+k0+1
    # C block: rows i=0,1, cols k=k0,k0+1
    #   K indices: k0*2+0, k0*2+1, (k0+1)*2+0, (k0+1)*2+1
    #
    # So for each pair of k values, we use 7 Strassen terms.
    # For the remaining single k, we use 4 schoolbook terms.
    #
    # Total rank: 7 * 7 + 4 = 53

    rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # Process pairs of k: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)
    for block in range(7):
        k0 = block * 2
        k1 = block * 2 + 1
        for r in range(7):
            # U is the same for all blocks
            for i in range(n):
                for j in range(m):
                    U[i * m + j, col] = strassen_U[i * m + j, r]

            # V: B block at cols k0, k1
            # B[j, k0] -> J = j*15 + k0, maps to Strassen B index j*2 + 0 = j*2
            # B[j, k1] -> J = j*15 + k1, maps to Strassen B index j*2 + 1
            for j in range(m):
                V[j * p + k0, col] = strassen_V[j * 2 + 0, r]
                V[j * p + k1, col] = strassen_V[j * 2 + 1, r]

            # W: C block at cols k0, k1
            # C[i, k0] -> K = k0*2 + i, maps to Strassen C index 0*2 + i = i
            # C[i, k1] -> K = k1*2 + i, maps to Strassen C index 1*2 + i
            for i in range(n):
                W[k0 * n + i, col] = strassen_W[0 * 2 + i, r]
                W[k1 * n + i, col] = strassen_W[1 * 2 + i, r]

            col += 1

    # Process remaining k=14 with schoolbook (4 terms)
    k_rem = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + k_rem, col] = 1
            W[k_rem * n + i, col] = 1
            col += 1

    assert col == rank, f"Column mismatch: {col} vs {rank}"

    # Verify the full decomposition
    def verify_full(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    val = 0.0
                    for c in range(rank):
                        val += U_mat[I, c] * V_mat[J, c] * W_mat[K, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check invalid positions
        for I in range(n * m):
            for J in range(m * p):
                for K in range(p * n):
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if I == i*m+j and J == j*p+k and K == k*n+i:
                                    valid = True
                    if not valid:
                        val = 0.0
                        for c in range(rank):
                            val += U_mat[I, c] * V_mat[J, c] * W_mat[K, c]
                        if abs(val) > 1e-9:
                            return False
        return True

    if verify_full(U, V, W):
        pass
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = rank_schoolbook

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
