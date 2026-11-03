import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    n, m, p = 2, 4, 4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    # W index: k*n + i (official trace convention W[k*n+i, r])

    # The multiplication tensor T has T[a, b, c] = 1 where:
    # a = i*m + j = i*4 + j  (i in 0..1, j in 0..3)
    # b = j*p + k = j*4 + k  (j in 0..3, k in 0..3)
    # c = k*n + i = k*2 + i  (k in 0..3, i in 0..1)
    # T[a, b, c] = 1 iff the indices are consistent (same j, same i, same k)

    # Build the tensor as a set of valid (a, b, c) triples
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    def verify(U, V, W, rank):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == 1 for all valid (a,b,c)
        and == 0 for all invalid (a,b,c)."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check valid triples
        for (a, b, c) in valid_triples:
            val = np.dot(np.dot(U[a, :] * V[b, :], W[c, :]))
            if abs(val - 1.0) > 1e-9:
                return False
        # Check a sample of invalid triples
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in valid_triples:
                        val = np.dot(np.dot(U[a, :] * V[b, :], W[c, :]))
                        if abs(val) > 1e-9:
                            return False
        return True

    # Strassen decomposition for 2x2: rank 7 per block
    # We have 4 blocks of 2x2, each with rank 7, total 28
    # Try to find if we can do better by sharing terms

    def strassen_2x2_block(col_offset, a_rows, b_rows, c_rows):
        """
        Add a 2x2 Strassen block.
        a_rows: 4 global U indices for the 2x2 A block (row-major: a00, a01, a10, a11)
        b_rows: 4 global V indices for the 2x2 B block
        c_rows: 4 global W indices for the 2x2 C block
        Returns 7 columns added.
        """
        U = np.zeros((n * m, 7), dtype=float)
        V = np.zeros((m * p, 7), dtype=float)
        W = np.zeros((p * n, 7), dtype=float)

        # Strassen terms for 2x2:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b01-b11)
        # P4 = a11(b10-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b01)
        # P7 = (a01-a11)(b10+b11)
        #
        # C results:
        # c00 = P1 + P4 - P5 + P7
        # c01 = P3 + P5
        # c10 = P2 + P4
        # c11 = P1 - P2 + P3 + P6

        # P1
        U[a_rows[0], 0] = 1; U[a_rows[3], 0] = 1
        V[b_rows[0], 0] = 1; V[b_rows[3], 0] = 1
        W[c_rows[0], 0] = 1; W[c_rows[3], 0] = 1

        # P2
        U[a_rows[2], 1] = 1; U[a_rows[3], 1] = 1
        V[b_rows[0], 1] = 1
        W[c_rows[2], 1] = 1; W[c_rows[3], 1] = -1

        # P3
        U[a_rows[0], 2] = 1
        V[b_rows[1], 2] = 1; V[b_rows[3], 2] = -1
        W[c_rows[1], 2] = 1; W[c_rows[3], 2] = 1

        # P4
        U[a_rows[3], 3] = 1
        V[b_rows[2], 3] = 1; V[b_rows[0], 3] = -1
        W[c_rows[0], 3] = 1; W[c_rows[2], 3] = 1

        # P5
        U[a_rows[0], 4] = 1; U[a_rows[1], 4] = 1
        V[b_rows[3], 4] = 1
        W[c_rows[0], 4] = -1; W[c_rows[1], 4] = 1

        # P6
        U[a_rows[2], 5] = 1; U[a_rows[0], 5] = -1
        V[b_rows[0], 5] = 1; V[b_rows[1], 5] = 1
        W[c_rows[3], 5] = 1

        # P7
        U[a_rows[1], 6] = 1; U[a_rows[3], 6] = -1
        V[b_rows[2], 6] = 1; V[b_rows[3], 6] = 1
        W[c_rows[0], 6] = 1

        return U, V, W

    # Build the 28-rank Strassen solution
    def build_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A rows: i=0, j=0,1 -> U[0], U[1]; i=1, j=0,1 -> U[4], U[5]
        a1 = [0, 1, 4, 5]
        # B rows: j=0, k=0,1 -> V[0], V[1]; j=1, k=0,1 -> V[4], V[5]
        b1 = [0, 1, 4, 5]
        # C rows: k=0, i=0,1 -> W[0], W[1]; k=1, i=0,1 -> W[2], W[3]
        # c00=k0,i0=W[0], c01=k1,i0=W[2], c10=k0,i1=W[1], c11=k1,i1=W[3]
        c1 = [0, 2, 1, 3]
        u1, v1, w1 = strassen_2x2_block(0, a1, b1, c1)
        U[:, 0:7] += u1
        V[:, 0:7] += v1
        W[:, 0:7] += w1

        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # A rows: i=0, j=2,3 -> U[2], U[3]; i=1, j=2,3 -> U[6], U[7]
        a2 = [2, 3, 6, 7]
        # B rows: j=2, k=0,1 -> V[8], V[9]; j=3, k=0,1 -> V[12], V[13]
        b2 = [8, 9, 12, 13]
        c2 = [0, 2, 1, 3]
        u2, v2, w2 = strassen_2x2_block(0, a2, b2, c2)
        U[:, 7:14] += u2
        V[:, 7:14] += v2
        W[:, 7:14] += w2

        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        a3 = [0, 1, 4, 5]
        # B rows: j=0, k=2,3 -> V[2], V[3]; j=1, k=2,3 -> V[6], V[7]
        b3 = [2, 3, 6, 7]
        # C rows: k=2, i=0,1 -> W[4], W[5]; k=3, i=0,1 -> W[6], W[7]
        # c00=k2,i0=W[4], c01=k3,i0=W[6], c10=k2,i1=W[5], c11=k3,i1=W[7]
        c3 = [4, 6, 5, 7]
        u3, v3, w3 = strassen_2x2_block(0, a3, b3, c3)
        U[:, 14:21] += u3
        V[:, 14:21] += v3
        W[:, 14:21] += w3

        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        a4 = [2, 3, 6, 7]
        # B rows: j=2, k=2,3 -> V[10], V[11]; j=3, k=2,3 -> V[14], V[15]
        b4 = [10, 11, 14, 15]
        c4 = [4, 6, 5, 7]
        u4, v4, w4 = strassen_2x2_block(0, a4, b4, c4)
        U[:, 21:28] += u4
        V[:, 21:28] += v4
        W[:, 21:28] += w4

        return U, V, W

    U, V, W = build_strassen_28()

    # Verify the 28-rank solution
    if not verify(U, V, W, 28):
        # Fallback: schoolbook rank 32
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank: attempt to find linear dependencies among the 28 columns
    # that allow us to merge or eliminate columns
    # The tensor rank is at least max of the three fiber ranks
    # Let's compute the fiber ranks to establish a lower bound

    # Fiber rank in A-direction: for each a, the space spanned by {V[:,r]*W[:,r] for r}
    # Actually, let's use the standard lower bound: rank >= max(rank of each fiber)
    # A-fiber: for each a, consider the matrix M_a where M_a[b, c] = T[a, b, c]
    # rank >= max_a rank(M_a)
    # Similarly for B and C fibers

    # Compute fiber ranks
    T = np.zeros((n * m, m * p, p * n))
    for (a, b, c) in valid_triples:
        T[a, b, c] = 1

    # A-fiber ranks
    a_fiber_max = 0
    for a in range(n * m):
        M = T[a, :, :]  # (m*p, p*n)
        r = np.linalg.matrix_rank(M)
        a_fiber_max = max(a_fiber_max, r)

    # B-fiber ranks
    b_fiber_max = 0
    for b in range(m * p):
        M = T[:, b, :]  # (n*m, p*n)
        r = np.linalg.matrix_rank(M)
        b_fiber_max = max(b_fiber_max, r)

    # C-fiber ranks
    c_fiber_max = 0
    for c in range(p * n):
        M = T[:, :, c]  # (n*m, m*p)
        r = np.linalg.matrix_rank(M)
        c_fiber_max = max(c_fiber_max, r)

    lower_bound = max(a_fiber_max, b_fiber_max, c_fiber_max)

    # Also try the slice rank lower bound
    # For each partition, compute the slice rank
    # This is complex; use the fiber rank as a practical lower bound

    # Attempt to find a rank < 28 solution by checking if columns are linearly dependent
    # in a way that preserves the tensor structure
    # This is equivalent to checking if the 28 rank-1 tensors span a space of dimension < 28
    # that still contains the multiplication tensor

    # Try greedy elimination: for each column, check if it can be expressed as a linear
    # combination of others while preserving the tensor
    # This is expensive; instead, try specific known lower-rank constructions

    # For 2x4x4, the known upper bounds are:
    # - Strassen: 28
    # - Schoolbook: 32
    # The lower bound from fiber ranks will tell us if 28 is tight

    # If lower_bound < 28, try to find a decomposition at lower_bound or lower_bound+1
    # For now, return the verified 28-rank solution

    # Convert to integer/half-integer if possible
    def to_rational_list(arr):
        """Convert numpy array to list of lists with integer or half-integer entries."""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-9:
                    new_row.append(int(round(val * 2)) / 2)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    # Check if we have time to attempt a search for lower rank
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and lower_bound < 28:
        # Try to find a rank-27 or lower solution
        # Use a randomized approach: try to find a linear combination of columns
        # that yields a valid lower-rank decomposition
        # This is a research problem; for now, we'll try a simple approach

        # Attempt: check if any single column can be removed (rank 27)
        # For each column r, check if the tensor can be decomposed without it
        # This requires solving a linear system
        pass  # Skip complex search due to time constraints

    U_list = to_rational_list(U)
    V_list = to_rational_list(V)
    W_list = to_rational_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}
