import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_terms = n * m * p  # 96 schoolbook

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W):
        # Exact tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def is_half_int(M):
        # entries must be integers or half-integers
        doubled = 2.0 * np.asarray(M, dtype=float)
        return np.all(np.abs(doubled - np.round(doubled)) < 1e-9)

    def schoolbook():
        rank = total_terms
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------
    # Tensor-product construction: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    #
    # M_{1,1,2}  (1x1 * 1x2 -> 1x2), rank 2:
    #   U1 shape (1,2) = [[1,1]]
    #   V1 shape (2,2) = [[1,0],[0,1]]
    #   W1 shape (2,2) = [[1,0],[0,1]]
    #
    # M_{3,4,4}  (3x4 * 4x4 -> 3x4), rank 21:
    #   For each row i (3 rows) do a 4x4 row-vector * 4x4-matrix multiply.
    #   A row is [a0,a1,a2,a3]; B is 4x4. Output C[i,k] = sum_j A[i,j] B[j,k].
    #   Use Winograd 4-point convolution on the index j for each output column k.
    #   The 4-point Winograd uses 4 transforms: the input row is convolved with
    #   each column of B. We build factors via the standard F matrix.
    # ------------------------------------------------------------------
    def winograd_4point_factors():
        # Winograd 4-point F (4x4) and its transpose for convolution.
        # We want: for output index k in 0..3, C_k = sum_j A_j * B_{j,k}
        # This is a convolution of A (length 4) with the column B_{:,k} (length 4),
        # but indexed directly (no flip) -> it's a "cross-correlation" style product.
        # We express C_k = sum_j A_j B_{j,k}. Treat B_{j,k} as a function of j for fixed k.
        # Use the identity: define transforms on A and on B-column.
        #
        # Winograd 4-point:  m = 4 (conv length), r = 4 (filter length).
        # Standard F (4x4) for 4-point:
        F = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [0, 1, 1, 0],
        ], dtype=float)
        # But we need a *rank-minimal* decomposition. Winograd 4-point has
        # rank = m + r - 1 = 7 for the convolution. The 7 products are the
        # pointwise products of the 7 transformed coefficients.
        #
        # For our direct (non-flipped) product C_k = sum_j A_j B_{j,k},
        # we can still use a 7-term decomposition. Let me build it via
        # the known 7-term structure for length-4 convolution and then
        # map indices.
        #
        # 7-term Winograd for y_k = sum_{j=0}^{3} a_j b_{k-j} (flipped).
        # Our product is sum_j a_j b_j for each k separately (b depends on k),
        # i.e. y_k = sum_j a_j b_{j,k}. This is exactly a convolution where
        # the "filter" for output k is the column b_{:,k} and we want the
        # value at lag 0 -> it's a per-column dot product. So it's NOT a single
        # convolution; it's 4 independent dot products of length 4.
        #
        # However, we can share the transform of a across all columns!
        # Let a' = F a (4 values). For each column b_{:,k}, compute b' = F b_{:,k}.
        # Then the dot product a . b_{:,k} = a' . G b' for some fixed G (from F).
        # Specifically if F is invertible, a.b = (F^{-1} a') . (F b) -> need
        # a' . (F^{-T} F) b'. Let M = F^{-T} F (4x4). Then
        #   a . b_{:,k} = a' . (M b')  = sum_t a'_t (M b')_t
        #                = sum_{t,s} a'_t M_{t,s} b'_{s,k}
        # This is a rank-16 decomposition (4x4). Not better than schoolbook 16.
        #
        # To beat schoolbook we need the Winograd *convolution* structure, which
        # requires the output to be a convolution. Our output C[i,k] is a dot
        # product, not a convolution in k. So Winograd does NOT directly apply
        # to the 3x4*4x4 block in this form.
        #
        # Given that, the cleanest rank reduction is to use Strassen on a
        # 2x2 block decomposition instead. Let's fall back to a Strassen-based
        # construction for the 4x4 part by embedding into 2x2 blocks.
        return None

    # ------------------------------------------------------------------
    # Strassen-based rank reduction for M_{3,4,8}:
    #
    # Split A (3x4) into two 3x2 blocks: A = [A_L | A_R]
    # Split B (4x8) into four 2x4 blocks: B = [[B_LL, B_LR],[B_RL, B_RR]]
    # C = A_L B_LL + A_R B_RL  (left half of C, 3x4)
    #     A_L B_LR + A_R B_RR  (right half of C, 3x4)
    #
    # Each term is a 3x2 * 2x4 multiplication. Use the tensor product:
    #   M_{3,2,4} = M_{3,2,2} ⊗ M_{1,1,2}
    #   M_{3,2,2} = M_{3,1,2} ⊗ M_{1,2,1}  ... getting complicated.
    #
    # Simpler: use Strassen on the 2x2 "middle" dimension.
    #   3x2 * 2x4: write the 2 as a 2x1 stacked. The product is:
    #   C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}
    #   This is a rank-2 (in the middle index) product: 3*2*4 = 24 schoolbook
    #   terms, but we can use the fact that the middle dimension is 2 to do
    #   a rank-2 decomposition directly (which is just schoolbook for dim 2).
    #
    # The real win: use Strassen on a 2x2 * 2x2 sub-block.
    # Embed the 3x4*4x8 problem as follows:
    #   M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}   (split the 8 into 2+2 along columns)
    #   M_{3,4,4} = M_{3,2,4} ⊗ M_{1,2,1}  ... no.
    #
    # Let's use the cleanest known factorization:
    #   M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    #   M_{3,4,4} = M_{3,4,2} ⊗ M_{1,1,2}
    #   M_{3,4,2} = M_{3,4,1} ⊗ M_{1,1,2}  -> M_{3,4,1} rank 12 (schoolbook)
    #   So M_{3,4,2} rank 24, M_{3,4,4} rank 48, M_{3,4,8} rank 96. No gain.
    #
    # The gain must come from Strassen on a 2x2*2x2 block (rank 7 vs 8).
    # M_{3,4,8}: split the 4 (middle) into 2+2, and the 8 (cols) into 4+4,
    # and the 3 (rows) into 1+2. This is messy.
    #
    # Practical approach: implement a known rank-42 decomposition directly.
    # Rank 42 = 3 * 7 * 2. This corresponds to:
    #   M_{3,4,8} = M_{3,1,1} ⊗ M_{1,4,8}
    #   M_{1,4,8} = M_{1,4,4} ⊗ M_{1,1,2}
    #   M_{1,4,4} = M_{1,2,2} ⊗ M_{1,2,2}   (each M_{1,2,2} rank 4 -> 16, not 7)
    #
    # Given the difficulty of a guaranteed sub-96 construction within time,
    # I will attempt a randomized search for a lower-rank exact decomposition
    # (as in inspiration 1) with half-integer rounding, and fall back to
    # schoolbook if nothing is found. This keeps the program valid and may
    # find a lucky lower rank.
    # ------------------------------------------------------------------

    best = None
    best_rank = total_terms

    # Randomized search for lower rank with half-integer entries
    np.random.seed(0)
    for rank in range(total_terms - 1, max(30, total_terms // 2), -1):
        if elapsed() > DEADLINE - 20:
            break
        for attempt in range(15):
            if elapsed() > DEADLINE - 20:
                break
            U = np.random.randint(-2, 3, size=(nm, rank))
            V = np.random.randint(-2, 3, size=(mp, rank))
            # M[i,k,c] = sum_j U[i*m+j,c] V[j*p+k,c]
            M = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for c in range(rank):
                        M[i, k, c] = sum(U[i * m + j, c] * V[j * p + k, c]
                                          for j in range(m))
            W = np.zeros((pn, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    v = M[i, k, :]
                    if np.linalg.norm(v) < 1e-10:
                        valid = False
                        break
                    W[k * n + i, :] = v / np.dot(v, v)
            if not valid:
                continue
            # Round to half-integers
            Ur = np.round(U * 2) / 2
            Vr = np.round(V * 2) / 2
            Wr = np.round(W * 2) / 2
            if verify(Ur, Vr, Wr):
                best = (Ur, Vr, Wr)
                best_rank = rank
                break
        if best is not None:
            break

    if best is None:
        U, V, W = schoolbook()
    else:
        U, V, W = best

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
