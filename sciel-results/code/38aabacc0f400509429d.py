import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 150s kill

    # Strassen 2x2 rank-7 factors (standard)
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

    # Kronecker product gives rank-49 for 4x4
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)

    # The Kronecker product of two 2x2 Strassen decompositions gives a 4x4 decomposition
    # but with a specific index ordering. We need to map from the Kronecker index ordering
    # to the standard row-major ordering.
    #
    # In the Kronecker product, the index for U is (i2, j2, i1, j1) where
    # i2,j2 are the outer 2x2 indices and i1,j1 are the inner 2x2 indices.
    # The Kronecker index is (i2*2+i1)*4 + (j2*2+j1) for a 4x4 matrix indexed (I,J)
    # where I = i2*2+i1, J = j2*2+j1.
    #
    # The standard row-major index for A[I,J] is I*4+J.
    # So the Kronecker index is already I*4+J where I = i2*2+i1, J = j2*2+j1.
    #
    # Wait, let me think more carefully.
    # np.kron(A, B) where A is (4,7) and B is (4,7) gives (16, 49).
    # The row index of the result is i*4+j where i is row in A, j is row in B.
    # So row = i*4 + j, where i in [0,4), j in [0,4).
    # This corresponds to a 4x4 matrix with row i, col j.
    #
    # For the 2x2 Strassen, U2 has rows indexed by (i,j) for i,j in {0,1}.
    # For 4x4, we split the 4x4 matrix into 2x2 blocks of 2x2 matrices.
    # The block (I,J) of the 4x4 matrix (where I,J in {0,1}) is a 2x2 matrix.
    # Element (i,j) of block (I,J) is at position (I*2+i, J*2+j) in the 4x4 matrix.
    # Row-major index: (I*2+i)*4 + (J*2+j) = I*8 + i*4 + J*2 + j.
    #
    # In np.kron(U2, U2), the row index is i2*4 + i1 where i2 is the row in the first U2
    # and i1 is the row in the second U2. But U2's rows are indexed by (a,b) for a,b in {0,1},
    # with row index a*2+b.
    #
    # So np.kron(U2, U2) row index = (a*2+b)*4 + (c*2+d) = a*8 + b*4 + c*2 + d.
    # This should correspond to the 4x4 index (I*2+i, J*2+j) where:
    # I = a, i = b, J = c, j = d? Let's check:
    # (I*2+i)*4 + (J*2+j) = (a*2+b)*4 + (c*2+d) = a*8+b*4+c*2+d. Yes!
    #
    # So the Kronecker product row index directly corresponds to the standard row-major
    # 4x4 index with I=a, i=b, J=c, j=d.
    #
    # But wait - we need to verify the column (rank) indexing is consistent across U, V, W.
    # In np.kron(U2, U2), the column index is c2*7 + c1 where c2 is col in first U2, c1 in second.
    # This is the same for V4 and W4 since they all use np.kron with 7-column factors.
    # So the rank-49 indexing is consistent.
    #
    # Let's verify the decomposition directly.

    # Verify: for all i,j,k in {0,1,2,3}, sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] == 1
    # And for invalid triples, the sum should be 0.

    def verify(U, V, W, n=4, m=4, p=4):
        rank = U.shape[1]
        # Check valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        # Check a sample of invalid triples
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    # Extract indices
                    i_u = u // m
                    j_u = u % m
                    j_v = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    # Valid if j_u == j_v and k_v == k_w and i_w == i_u
                    if not (j_u == j_v and k_v == k_w and i_w == i_u):
                        s = 0
                        for r in range(rank):
                            s += U[u, r] * V[v, r] * W[w, r]
                        if s != 0:
                            return False
        return True

    # Verify the Kronecker product decomposition
    if verify(U4, V4, W4):
        if time.time() - start_time > DEADLINE:
            # Should not happen, but fallback
            n, m, p = 4, 4, 4
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }

    # If verification failed, try an alternative permutation approach
    # The issue might be with how np.kron orders the indices.
    # Let's try to construct the 4x4 decomposition from 2x2 blocks explicitly.

    # Alternative: build 4x4 from 2x2 Strassen blocks manually
    # For 4x4 matrix multiplication, split into 2x2 blocks of 2x2 matrices.
    # C = A*B where A, B, C are 4x4.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # Similarly for B and C.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each 2x2 multiplication uses 7 Strassen terms.
    # For C00: need A00*B00 (7 terms) + A01*B10 (7 terms) = 14 terms
    # Similarly for C01, C10, C11: 14 terms each
    # Total: 56 terms. But we can share some? No, the matrices are different.
    # Actually, this gives rank 56, not 49.
    #
    # The Kronecker product approach should give 49 if the indexing is right.
    # Let me try a different permutation.

    # Try: maybe the issue is that np.kron orders differently than expected.
    # Let's try all permutations of the 16 rows and see if any work.
    # Actually, let's just try a few specific permutations.

    # Permutation 1: identity (already tried, should work)
    # Permutation 2: swap the two 2x2 block levels
    perm2 = np.zeros(16, dtype=int)
    for I in range(2):
        for i in range(2):
            for J in range(2):
                for j in range(2):
                    std_idx = (I*2+i)*4 + (J*2+j)
                    # Swap block and element indices: treat as (i,I,j,J) instead of (I,i,J,j)
                    alt_idx = (i*2+I)*4 + (j*2+J)
                    perm2[std_idx] = alt_idx

    if verify(U4[perm2, :], V4[perm2, :], W4[perm2, :]):
        if time.time() - start_time > DEADLINE:
            n, m, p = 4, 4, 4
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        return {
            "U": U4[perm2, :].tolist(),
            "V": V4[perm2, :].tolist(),
            "W": W4[perm2, :].tolist()
        }

    # Try permutation 3: reverse bit order
    perm3 = np.zeros(16, dtype=int)
    for idx in range(16):
        b0 = (idx >> 0) & 1
        b1 = (idx >> 1) & 1
        b2 = (idx >> 2) & 1
        b3 = (idx >> 3) & 1
        # Reverse: b3 b2 b1 b0
        new_idx = b3 * 8 + b2 * 4 + b1 * 2 + b0
        perm3[idx] = new_idx

    if verify(U4[perm3, :], V4[perm3, :], W4[perm3, :]):
        if time.time() - start_time > DEADLINE:
            n, m, p = 4, 4, 4
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        return {
            "U": U4[perm3, :].tolist(),
            "V": V4[perm3, :].tolist(),
            "W": W4[perm3, :].tolist()
        }

    # Fallback: schoolbook rank 64
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
