"""Program searching for low-rank exact tensor decomposition of 2x3 by 3x8 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    deadline = time.time() + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    base_rank = n * m * p

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-9, rtol=1e-9)

    def tensor_residual(U, V, W):
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T - computed
        return np.sum(diff * diff)

    # ---------- Schoolbook baseline ----------
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    best = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist(), "rank": base_rank}

    # ---------- Structured block (Kronecker) candidates ----------
    # A full Kronecker product (A⊗B)⊗(C⊗D)⊗(E⊗F) exactly decomposes the
    # block-tensor whose (block-row, block-col) sub-block is the product of the
    # corresponding sub-matrices. We therefore enumerate a partition of the 8
    # columns into groups so that the grouped 2×m by m×(sum) multiplication is
    # exactly the Kronecker product of the per-group multiplications, and build
    # a rank-multiplied factor. We verify every candidate exactly.

    def build_kronecker(groups, factors):
        """groups: list of column-index lists (partition of range(p)).
        factors: list of (U_g, V_g, W_g) for each group (multiplication
                 of shape n x m x len(group)).
        Returns (U, V, W) of full size via block-Kronecker expansion, or None."""
        # Full U: rows (i*m + j) for i in [0,n), j in [0,m); cols = product of group ranks
        # V: rows (j*p + k); W: rows (k*n + i).
        # For a Kronecker product over groups g with local dims (n_g, m_g, p_g):
        #   U_full row (i*m+j) col (prod index) = prod_g U_g[i*m_g + j, idx_g]
        #   V_full row (j*p+k) col = prod_g V_g[j*p_g + k_g, idx_g]  (k_g = local col in group g)
        #   W_full row (k*n+i) col = prod_g W_g[k_g*n_g + i, idx_g]
        # This is valid when the full multiplication is the Kronecker product of the
        # group multiplications, i.e. T[i*m+j, j*p+k, k*n+i] = prod_g T_g[...].
        # We will simply construct it and verify; if it fails we discard.

        granks = [f[0].shape[1] for f in factors]
        total_rank = int(np.prod(granks)) if granks else 0
        if total_rank == 0:
            return None

        # Precompute column-index enumeration
        import itertools
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))

        for col, idxs in enumerate(itertools.product(*[range(r) for r in granks])):
            uval = 1.0
            vval = 1.0
            wval = 1.0
            # We need to map each group's local (i,j), (j,k_g), (k_g,i) to global.
            # But the Kronecker structure couples i,j,k across groups via the SAME i,j.
            # The product over groups of T_g[i*m_g + j, j*p_g + k_g, k_g*n_g + i] equals
            # prod_g T_g[...] which is 1 iff every group's (i,j,k_g) is a valid triple,
            # i.e. iff (i,j,k) with k = sum over groups of (k_g * stride) is valid.
            # This only matches T for a specific grouping of k. We compute the value
            # directly by checking the product of the per-group tensors.
            ok = True
            for g, (U_g, V_g, W_g) in enumerate(factors):
                idx = idxs[g]
                # For a fixed global (i, j) and local k_g, the group tensor entry is:
                # T_g[i*m_g + j, j*p_g + k_g, k_g*n_g + i]
                # We don't know i,j,k_g here without iterating; instead we compute the
                # full product tensor and check against T. Simpler: build via einsum of
                # the Kronecker factors and verify.
            # Because direct per-column mapping needs i,j,k_g, we instead build the
            # factor matrices using the Kronecker product of the local factors:
            # U_full = kron over g of U_g (with appropriate row/col reshaping)
            pass

        # Simpler, correct construction: U_full is the Kronecker product of U_g across
        # groups, with rows indexed by the tuple of local row indices and columns by the
        # tuple of local column indices. We reshape each U_g to (n_g, m_g, r_g) and
        # take the tensor Kronecker product over groups, then reshape rows to (n*m) and
        # columns to total_rank. This is exactly the block-Kronecker expansion.
        # We do this via repeated np.kron on reshaped matrices.

        def kron_rows(A, B, a_rows, b_rows):
            # A: (a_rows, rA), B: (b_rows, rB) -> (a_rows*b_rows, rA*rB) via kron
            rA = A.shape[1]; rB = B.shape[1]
            return np.kron(A, B).reshape(a_rows * b_rows, rA * rB)

        # Row sizes per group for U: n*m (same for all groups since n,m shared)
        # V row size: m * p_g ; W row size: p_g * n
        U_full = None
        V_full = None
        W_full = None
        for g, (U_g, V_g, W_g) in enumerate(factors):
            if U_full is None:
                U_full = U_g.copy()
                V_full = V_g.copy()
                W_full = W_g.copy()
                u_rows = n * m
                v_rows = m * len(groups[g])
                w_rows = len(groups[g]) * n
            else:
                U_full = kron_rows(U_full, U_g, u_rows, n * m)
                V_full = kron_rows(V_full, V_g, v_rows, m * len(groups[g]))
                W_full = kron_rows(W_full, W_g, w_rows, len(groups[g]) * n)
                u_rows *= n * m
                v_rows *= m * len(groups[g])
                w_rows *= len(groups[g]) * n
        # u_rows should be (n*m)^(num_groups) which is NOT n*m. This Kronecker
        # expansion over groups does NOT preserve the row count unless num_groups==1.
        # Therefore a multi-group Kronecker over the SAME (i,j) is not the right
        # construction. The valid Kronecker is over the column index only.
        return None

    # The multi-group Kronecker over (i,j) is invalid (row count explodes). The
    # structurally valid Kronecker is over the column (k) index: treat the 8 columns
    # as a product of two 4-column halves, i.e. p = 4*2. Then the multiplication
    # (2x3) x (3x8) with k = 4*k2 + k1 (k2 in [0,4), k1 in [0,2)) is the Kronecker
    # product of (2x3)x(3x4) [over k2] and (2x3)x(3x2) [over k1] ONLY IF the two
    # sub-multiplications share the same (A, B) and the result is C[i, 4*k2+k1] =
    # C2[i, k2] * C1[i, k1]. That is NOT true for general matrix multiplication.
    # So a pure Kronecker factorization of the multiplication tensor is generally
    # impossible; the only exact Kronecker is the trivial single-block one.
    #
    # Hence the structured idea reduces to: search for a low-rank factor of a
    # smaller sub-multiplication and use it as a *block* factor where the block
    # structure is a direct sum (block-diagonal) over independent column groups.
    # A block-diagonal (direct-sum) construction is valid: if we split columns into
    # groups and each group's multiplication is decomposed independently, the full
    # tensor is the direct sum, and rank adds. This is just the schoolbook applied
    # per group, giving rank = sum of group ranks = n*m*p again. No improvement.
    #
    # Conclusion: for 2x3x8 no exact structured block factorization beats schoolbook
    # via Kronecker/direct-sum. We therefore rely on the gradient search below,
    # but seed it with the schoolbook factors (a warm start) so it can explore
    # nearby low-rank directions, and we add a final high-precision polish.

    # ---------- Gradient descent with warm start ----------
    def gradient_descent(U, V, W, lr=0.01, max_iter=500, tol=1e-12):
        U = U.copy(); V = V.copy(); W = W.copy()
        r = U.shape[1]
        prev_loss = tensor_residual(U, V, W)
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if prev_loss < tol:
                break
            computed = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = computed - T
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            new_loss = tensor_residual(U, V, W)
            if new_loss < prev_loss * 0.999:
                lr = min(lr * 1.05, 0.1)
            else:
                lr = max(lr * 0.95, 1e-6)
            prev_loss = new_loss
        return U, V, W, prev_loss

    def polish(U, V, W, max_iter=2000, tol=1e-14):
        """High-precision polish: very small lr, many iterations."""
        U = U.copy(); V = V.copy(); W = W.copy()
        prev_loss = tensor_residual(U, V, W)
        lr = 1e-4
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if prev_loss < tol:
                break
            computed = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = computed - T
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            new_loss = tensor_residual(U, V, W)
            if new_loss < prev_loss:
                lr = min(lr * 1.02, 1e-3)
            else:
                lr = max(lr * 0.9, 1e-7)
            prev_loss = new_loss
        return U, V, W, prev_loss

    def try_rank(r, rng, max_restarts=3, warm=False):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            if warm:
                # Warm start: take first r columns of schoolbook and add small noise
                U = U0[:, :r].copy() + rng.normal(size=(n * m, r)) * 0.1
                V = V0[:, :r].copy() + rng.normal(size=(m * p, r)) * 0.1
                W = W0[:, :r].copy() + rng.normal(size=(p * n, r)) * 0.1
            else:
                scale = 1.0 / np.sqrt(r)
                U = rng.normal(size=(n * m, r)) * scale
                V = rng.normal(size=(m * p, r)) * scale
                W = rng.normal(size=(p * n, r)) * scale
            U, V, W, loss = gradient_descent(U, V, W, lr=0.05, max_iter=400, tol=1e-14)
            if loss < 1e-10:
                U, V, W, loss = polish(U, V, W)
                if loss < 1e-12:
                    return U, V, W
        return None

    rng = np.random.RandomState(SEED)
    min_possible = max(n * m, m * p, p * n)  # 24

    for r in range(min_possible, base_rank):
        if time.time() > deadline:
            break
        result = try_rank(r, rng, max_restarts=2, warm=(r >= min_possible))
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                # Round to half-integers if close, else return floats
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return schoolbook decomposition
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
