"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115  # safety margin under 120s (evaluator kills at 150s)

    size_a = n * m   # 4
    size_b = m * p   # 28
    size_c = p * n   # 28

    # True tensor: T[a,b,c]=1 where a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
        return bool(np.allclose(T, T_true, atol=atol))

    def schoolbook():
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_block():
        """7 independent 2x2 Strassen blocks -> rank 49 (clean, verified)."""
        num_blocks = p // 2  # 7
        rank = num_blocks * 7
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        # Strassen forms for A (indices A00=0,A01=1,A10=2,A11=3)
        U_forms = [
            [1, 0, 0, 1],   # M1 = A00 + A11
            [0, 0, 1, 1],   # M2 = A10 + A11
            [1, 0, 0, 0],   # M3 = A00
            [0, 0, 0, 1],   # M4 = A11
            [1, 1, 0, 0],   # M5 = A00 + A01
            [-1, 0, 1, 0],  # M6 = A10 - A00
            [0, 1, 0, -1],  # M7 = A01 - A11
        ]
        for r in range(num_blocks):
            idx_e = 2 * r            # B[0,2r]
            idx_f = 2 * r + 1        # B[0,2r+1]
            idx_g = 14 + 2 * r       # B[1,2r]
            idx_h = 14 + 2 * r + 1   # B[1,2r+1]
            w00 = 4 * r
            w10 = 4 * r + 1
            w01 = 4 * r + 2
            w11 = 4 * r + 3
            for s in range(7):
                c = 7 * r + s
                for i in range(4):
                    U[i, c] = U_forms[s][i]
                if s == 0:
                    V[idx_e, c] = 1; V[idx_h, c] = 1
                elif s == 1:
                    V[idx_e, c] = 1
                elif s == 2:
                    V[idx_f, c] = 1; V[idx_h, c] = -1
                elif s == 3:
                    V[idx_e, c] = -1; V[idx_g, c] = 1
                elif s == 4:
                    V[idx_h, c] = 1
                elif s == 5:
                    V[idx_e, c] = 1; V[idx_f, c] = 1
                else:
                    V[idx_g, c] = 1; V[idx_h, c] = 1
                if s == 0:
                    W[w00, c] = 1; W[w11, c] = 1
                elif s == 1:
                    W[w10, c] = 1; W[w11, c] = -1
                elif s == 2:
                    W[w01, c] = 1; W[w11, c] = 1
                elif s == 3:
                    W[w00, c] = 1; W[w10, c] = 1
                elif s == 4:
                    W[w00, c] = -1; W[w01, c] = 1
                elif s == 5:
                    W[w11, c] = 1
                else:
                    W[w00, c] = 1
        return U, V, W

    def try_cross_block_reduction(U0, V0, W0):
        """
        Exploit block structure: each of the 7 Strassen blocks is independent.
        For each block, the 7 rank-1 tensors span a 7-dim subspace.  Try to
        re-express the SAME block tensor using fewer rank-1 terms (a smaller
        decomposition of the 2x2x2 Strassen tensor) and, if found, apply it
        to every block -> total rank 7 * (smaller block rank).
        """
        if time.time() > deadline:
            return None
        U0 = np.asarray(U0, dtype=np.float64)
        V0 = np.asarray(V0, dtype=np.float64)
        W0 = np.asarray(W0, dtype=np.float64)
        R0 = U0.shape[1]
        num_blocks = p // 2
        per = R0 // num_blocks  # 7

        # For block 0, the 7 tensors live in the subspace of A-indices {0,1,2,3}
        # and B-indices {0,1,14,15} and C-indices {0,1,2,3}.
        # Build the 7x(4*4*4) matrix of block-0 tensors (flattened in that subspace).
        a_idx = [0, 1, 2, 3]
        b_idx = [0, 1, 14, 15]
        c_idx = [0, 1, 2, 3]
        B0 = np.zeros((per, 4 * 4 * 4), dtype=np.float64)
        for s in range(per):
            col = 7 * 0 + s
            for ai, a in enumerate(a_idx):
                for bi, b in enumerate(b_idx):
                    for ci, c in enumerate(c_idx):
                        B0[s, ai * 16 + bi * 4 + ci] = U0[a, col] * V0[b, col] * W0[c, col]
        # Numerical rank of the block tensor space
        try:
            svals = np.linalg.svd(B0, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9 * (svals[0] if len(svals) else 1.0)
        block_rank = int(np.sum(svals > tol))
        if block_rank >= per:
            # No compression of a single block via linear dependence.
            return None
        return None  # conservative: rank-1 re-expression not guaranteed

    def try_global_nullspace(U0, V0, W0):
        """
        Build the full rank x (4*28*28) matrix of the 49 rank-1 tensors and
        compute its numerical rank.  If the 49 tensors are linearly dependent
        (num_rank < 49), there is redundancy; however, turning linear
        dependence into a rank reduction requires a rank-1 re-expression,
        which is not automatic.  We report the dimension for awareness and
        attempt a greedy drop of any term that is (numerically) zero.
        """
        if time.time() > deadline:
            return None
        U0 = np.asarray(U0, dtype=np.float64)
        V0 = np.asarray(V0, dtype=np.float64)
        W0 = np.asarray(W0, dtype=np.float64)
        R0 = U0.shape[1]
        # Drop any exactly-zero tensor
        keep = []
        for r in range(R0):
            if np.any(np.abs(U0[:, r]) > 1e-12) and np.any(np.abs(V0[:, r]) > 1e-12) and np.any(np.abs(W0[:, r]) > 1e-12):
                keep.append(r)
        if len(keep) < R0:
            return (U0[:, keep], V0[:, keep], W0[:, keep])
        return None

    # --- main flow ---
    # 1) Constructive clean rank-49 Strassen
    U, V, W = strassen_block()
    best = (U, V, W)
    if verify(U, V, W):
        # 2) Attempt reductions (both return None unless a strict improvement)
        for fn in (try_global_nullspace, try_cross_block_reduction):
            if time.time() > deadline:
                break
            red = fn(U, V, W)
            if red is not None:
                Ur, Vr, Wr = red
                if verify(Ur, Vr, Wr) and Ur.shape[1] < best[0].shape[1]:
                    best = (Ur, Vr, Wr)
    else:
        # Fallback schoolbook if the constructive one somehow failed
        U, V, W = schoolbook()
        best = (U, V, W)

    Uf, Vf, Wf = best
    # Round to nearest half-integer for clean output
    Uf = np.round(Uf * 2) / 2.0
    Vf = np.round(Vf * 2) / 2.0
    Wf = np.round(Wf * 2) / 2.0
    if not verify(Uf, Vf, Wf, atol=1e-6):
        Uf, Vf, Wf = best

    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
# EVOLVE-BLOCK-END
