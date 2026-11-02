import numpy as np
import time

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 120  # seconds, evaluator kills after 150s

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    # Schoolbook fallback (rank 18)
    U_school = np.zeros((nm, 18))
    V_school = np.zeros((mp, 18))
    W_school = np.zeros((pn, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-9):
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False

        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=tol):
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification"""
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False

        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(s, 1.0, atol=tol):
                        return False
        return True

    # Try Winograd rank 8 first
    G = np.array([
        [1, 0, 0],
        [1, 1, 1],
        [1, -1, 1],
        [0, 1, -1]
    ], dtype=float)

    P = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, -1, 0],
        [0, 1, 0]
    ], dtype=float)

    AT = np.array([
        [1, 1, 1, 0],
        [0, 1, -1, -1],
        [1, 0, -1, 1]
    ], dtype=float)

    R = 8
    U_wino = np.zeros((nm, R))
    V_wino = np.zeros((mp, R))
    W_wino = np.zeros((pn, R))

    for i in range(n):
        for t in range(4):
            r = i * 4 + t
            for j in range(m):
                U_wino[i*m + j, r] = G[t, j]
                for k in range(p):
                    V_wino[j*p + k, r] = P[t, j]
            for k in range(p):
                W_wino[k*n + i, r] = AT[k, t]

    if verify_fast(U_wino, V_wino, W_wino):
        return {
            "U": U_wino.tolist(),
            "V": V_wino.tolist(),
            "W": W_wino.tolist()
        }

    # Winograd rank 8 doesn't work. Let's try the hybrid Strassen approach.
    # Split the 3x3 multiplication into:
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:2] @ B[2:3, :]
    # First term: 2x2 @ 2x3 = rank 9 (Strassen rank 7 for 2x2x2 + rank 2 for 2x2x1)
    # Second term: 2x1 @ 1x3 = rank 1 (outer product)
    # Total: rank 10

    # Actually let me think more carefully.
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0,1; j=0,1,2; k=0,1,2
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]

    # Let me try to build a rank 10 decomposition manually.
    # Using the identity:
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]

    # Group terms:
    # For fixed i,k:
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]

    # Try to factor into rank 10 terms.
    # One approach: use 6 terms for the "main" part and 4 for the cross terms.

    # Actually, let me try a direct construction.
    # We have 18 monomials: A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We want to write each as a sum of 10 rank-1 tensors.

    # Let me try the following structure:
    # For each (i,k), C[i,k] is a bilinear form in the 6 entries of A and 9 entries of B.
    # We need 18 such bilinear forms.

    # Alternative: try to find a rank-10 decomposition via the known results.
    # For 2x3x3, the tensor rank is known to be at most 10 and at least... let me just try.

    # Let me construct a rank 10 solution:
    # Terms 1-4: handle A[:, 0:2] @ B[0:2, :] using Strassen-like
    # Terms 5-6: handle A[:, 0:2] @ B[2:3, :] (2x2 @ 2x1 = rank 2)
    # Terms 7-8: handle A[:, 2:3] @ B[0:2, :] (2x1 @ 1x2 = rank 1... wait)
    # Terms 9-10: handle A[:, 2:3] @ B[2:3, :] (2x1 @ 1x1 = rank 1)

    # Actually:
    # A[:, 0:2] @ B[0:2, :] is 2x2 @ 2x2 = rank 7 (Strassen)
    # A[:, 0:2] @ B[2:3, :] is 2x2 @ 2x1 = rank 2
    # A[:, 2:3] @ B[0:2, :] is 2x1 @ 1x2 = rank 1
    # A[:, 2:3] @ B[2:3, :] is 2x1 @ 1x1 = rank 1
    # Total: 7 + 2 + 1 + 1 = 11

    # Hmm, that's 11. Can we do better?
    # A[:, 0:2] @ B[0:2, :] is 2x2 @ 2x3. Let me split B[0:2, :] into B[0:2, 0:2] and B[0:2, 2:3].
    # 2x2 @ 2x2 = rank 7
    # 2x2 @ 2x1 = rank 2
    # A[:, 2:3] @ B[:, :] is 2x1 @ 1x3 = rank 1
    # Total: 7 + 2 + 1 = 10!

    # So:
    # C = A[:, 0:2] @ B[0:2, 0:2] + A[:, 0:2] @ B[0:2, 2:3] + A[:, 2:3] @ B[:, :]
    # Rank: 7 + 2 + 1 = 10

    # Let me construct this.
    # Let A1 = A[:, 0:2] (2x2), B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2:3] (2x1), A2 = A[:, 2:3] (2x1), B2 = B[:, :] (3x3... wait no)

    # Actually A[:, 2:3] @ B[:, :] is (2x1) @ (3x3) = 2x3. But A[:, 2:3] only uses A[i,2].
    # So C[i,k] += A[i,2] * B[2,k] for all k.
    # That's A[i,2] * B[2,0] + A[i,2] * B[2,1] + A[i,2] * B[2,2]
    # This is rank 1 in the tensor sense? No, it's 3 separate monomials.
    # A[i,2]*B[2,0], A[i,2]*B[2,1], A[i,2]*B[2,2] - these are 3 rank-1 terms.
    # So total rank would be 7 + 2 + 3 = 12.

    # Hmm. Let me reconsider.
    # A[:, 2:3] @ B[2:3, :] is (2x1) @ (1x3) = 2x3 outer product.
    # C[i,k] += A[i,2] * B[2,k]
    # This is 3 monomials: A[i,2]*B[2,0], A[i,2]*B[2,1], A[i,2]*B[2,2]
    # Each is rank 1, so 3 rank-1 terms total.

    # And A[:, 0:2] @ B[0:2, :] is (2x2) @ (2x3) = 2x3.
    # This has 4*3 = 12 monomials: A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Rank of 2x2x3 tensor: it's the Kronecker product of 2x2x1 and 2x1x3? No.
    # 2x2x3 = (2x2x1) @ (1x2x3)? No.
    # The tensor for 2x2 @ 2x3: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is rank... for each i, it's a 2x3 matrix that's a sum of 2 rank-1 matrices.
    # The tensor rank of 2x2x3 is at most 2*2 = 4? No, that's not right.
    # Actually, the tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    # For 2x2 @ 2x3, the tensor has indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1,2}.
    # T[i,j,k] = 1 for all combinations.
    # This is the tensor for matrix multiplication (2,2,3).
    # Its rank is at most 2*3 = 6 (schoolbook), and at least...
    # Actually, I recall that the tensor rank of m x n x p matrix multiplication is at least max(mn, mp, np) and at most mn... no.
    # The rank is at least max(mn, np, mp) for some cases, but not always.

    # Let me just try to construct it directly.
    # For 2x2 @ 2x3:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]

    # Can I write this as a sum of rank-1 tensors with fewer than 6 terms?
    # Let me try rank 4:
    # T1: A[0,0]*B[0,0] -> U[0,0]=1, V[0,0]=1, W[0,0]=1
    # T2: A[0,1]*B[1,0] -> U[0,1]=1, V[1,0]=1, W[0,0]=1
    # T3: A[0,0]*B[0,1] -> U[0,0]=1, V[0,1]=1, W[0,1]=1
    # T4: A[0,1]*B[1,1] -> U[0,1]=1, V[1,1]=1, W[0,1]=1
    # ... this is getting complicated.

    # Let me just use schoolbook for the subproblems and see what total rank I get.
    # 2x2 @ 2x2: schoolbook rank 4
    # 2x2 @ 2x1: schoolbook rank 4
    # 2x1 @ 1x3: schoolbook rank 2
    # Total: 4 + 4 + 2 = 10

    # Wait, 2x1 @ 1x3: C[i,k] = A[i,0]*B[0,k] for i in {0,1}, k in {0,1,2}
    # That's 2*1*3 = 6 monomials. Schoolbook rank 6.
    # So total: 4 + 4 + 6 = 14.

    # Hmm. Let me just try the brute force approach with a time limit.
    # Actually, let me try the Winograd approach more carefully.

    # Winograd for 2x3 @ 3x3:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For each row i of A, we have a 1x3 @ 3x3 = 1x3 multiplication.
    # This is a vector (length 3) times a matrix (3x3).
    # C[i,:] = A[i,:] @ B
    # This is 3 scalar multiplications: C[i,k] = sum_j A[i,j]B[j,k]
    # For fixed i, this is 3 outputs, each a dot product of length 3.
    # The tensor for 1x3 @ 3x3 has rank at most 3 (one per output) if we fix A.
    # But we need it to work for all A.

    # Actually, for fixed i, C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear form in A[i,:] (3-vector) and B (3x3 = 9-vector).
    # The tensor T[j,k] = 1 for all j,k (since C[i,k] = sum_jk A[i,j]B[j,k])
    # Wait, T[j,k] for the inner product is just the identity in the j,k space.
    # The rank of the 3x3 identity tensor is 3.
    # So for each row i, we need rank 3.
    # For 2 rows, total rank 6? That can't be right because we need different terms for different i.

    # Let me think again. The full tensor has indices (i, j, k) with:
    # i in {0,1} (row of A, row of C)
    # j in {0,1,2} (col of A, row of B)
    # k in {0,1,2} (col of B, col of C)
    # T[i,j,k] = 1

    # This is a 2x3x3 tensor.
    # The rank of this tensor is what we're looking for.

    # Let me just try to construct a rank-10 solution and verify it.
    # If it doesn't work, fall back to schoolbook.

    # For now, let me return the schoolbook solution since the Winograd one didn't verify.
    # The rank is 18, which is the baseline.

    # Actually, let me try one more thing: a rank-12 construction.
    # Use 2x2 @ 2x2 (rank 4) + 2x2 @ 2x1 (rank 4) + 2x1 @ 1x3 (rank 4) = 12
    # Wait, 2x1 @ 1x3 is 2*1*3 = 6 monomials, schoolbook rank 6.
    # So 4 + 4 + 6 = 14.

    # Or: 2x2 @ 2x3 (rank ?) + 2x1 @ 1x3 (rank ?)
    # 2x2 @ 2x3: 12 monomials.
    # 2x1 @ 1x3: 6 monomials.
    # Total 18 monomials.

    # I think the best I can do without a clever construction is schoolbook rank 18.
    # Let me return that.

    if time.time() - start_time > TIME_LIMIT - 10:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
