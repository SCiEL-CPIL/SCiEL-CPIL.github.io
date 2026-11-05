"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _rank17():
    """Return the known, verified rank-17 decomposition (Strassen + direct)."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))

    terms = [
        # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _schoolbook():
    """Final fallback: exact schoolbook decomposition (rank n*m*p)."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return U_fb, V_fb, W_fb

def _try_rank16(deadline):
    """
    Search for a rank-16 decomposition.

    Key idea: a rank-16 tensor lives in the set of tensors whose
    16 component vectors (U[:,r], V[:,r], W[:,r]) satisfy the tensor
    equation. We warm-start from the known rank-17 by dropping one
    column, then refine using a column-wise least-squares / alternating
    projection scheme with multiple restarts so we can escape local
    minima. We accept the result only if it matches T exactly.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    Tflat = T.reshape(n_a * n_b * n_c)

    def error_of(U, V, W):
        res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum((res.reshape(-1) - Tflat) ** 2)

    def refine(U, V, W, max_sweeps, deadline):
        """Efficient alternating least-squares with incremental residual tracking."""
        # Precompute feature vectors for each rank component
        # For U update at rank r: feat_U[r] = V[:,r] ⊗ W[:,r]  (shape n_b*n_c)
        # For V update at rank r: feat_V[r] = U[:,r] ⊗ W[:,r]  (shape n_a*n_c)
        # For W update at rank r: feat_W[r] = U[:,r] ⊗ V[:,r]  (shape n_a*n_b)

        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break

            # Compute current full tensor product once per sweep
            full_prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

            # --- update U ---
            for r in range(rank):
                v = V[:, r]; w = W[:, r]
                # residual = T - full_prod + U[:,r] v w  (add back this term)
                res = full_prod.copy()
                res -= np.einsum('a,b,c->abc', U[:, r], v, w)
                feat = (v[:, None] * w[None, :]).reshape(-1)
                resA = res.reshape(n_a, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                U[:, r] = -(resA @ feat) / f2

            # Recompute full_prod after U updates
            full_prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

            # --- update V ---
            for r in range(rank):
                u = U[:, r]; w = W[:, r]
                res = full_prod.copy()
                res -= np.einsum('a,b,c->abc', u, V[:, r], w)
                feat = (u[:, None] * w[None, :]).reshape(-1)
                resB = res.reshape(n_b, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                V[:, r] = -(resB @ feat) / f2

            # Recompute full_prod after V updates
            full_prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

            # --- update W ---
            for r in range(rank):
                u = U[:, r]; v = V[:, r]
                res = full_prod.copy()
                res -= np.einsum('a,b,c->abc', u, v, W[:, r])
                feat = (u[:, None] * v[None, :]).reshape(-1)
                resC = res.reshape(n_c, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                W[:, r] = -(resC @ feat) / f2

        return U, V, W

    def polish(U, V, W, deadline, max_iters=200):
        """High-precision polish: alternate between exact column solves with tighter convergence."""
        for it in range(max_iters):
            if time.time() > deadline:
                break
            changed = False

            # Update U columns
            for r in range(rank):
                v = V[:, r]; w = W[:, r]
                full_prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                res = full_prod - np.einsum('a,b,c->abc', U[:, r], v, w)
                feat = (v[:, None] * w[None, :]).reshape(-1)
                resA = res.reshape(n_a, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                new_U_r = -(resA @ feat) / f2
                if np.max(np.abs(new_U_r - U[:, r])) > 1e-12:
                    U[:, r] = new_U_r
                    changed = True

            # Update V columns
            for r in range(rank):
                u = U[:, r]; w = W[:, r]
                full_prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                res = full_prod - np.einsum('a,b,c->abc', u, V[:, r], w)
                feat = (u[:, None] * w[None, :]).reshape(-1)
                resB = res.reshape(n_b, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                new_V_r = -(resB @ feat) / f2
                if np.max(np.abs(new_V_r - V[:, r])) > 1e-12:
                    V[:, r] = new_V_r
                    changed = True

            # Update W columns
            for r in range(rank):
                u = U[:, r]; v = V[:, r]
                full_prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                res = full_prod - np.einsum('a,b,c->abc', u, v, W[:, r])
                feat = (u[:, None] * v[None, :]).reshape(-1)
                resC = res.reshape(n_c, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                new_W_r = -(resC @ feat) / f2
                if np.max(np.abs(new_W_r - W[:, r])) > 1e-12:
                    W[:, r] = new_W_r
                    changed = True

            if not changed:
                break

        return U, V, W

    best = None
    best_err = np.inf
    base_U, base_V, base_W = _rank17()

    # Restart strategy: drop a different column from rank-17 each time.
    # Use varied perturbation scales for diversity.
    for restart in range(60):
        if time.time() > deadline - 8:
            break
        drop = restart % 17
        U = np.delete(base_U, drop, axis=1).copy()
        V = np.delete(base_V, drop, axis=1).copy()
        W = np.delete(base_W, drop, axis=1).copy()

        # Varied perturbation: small for early restarts, larger for later
        if restart >= 17:
            scale = 0.01 * (1.0 + 0.5 * np.sin(restart * 0.7))
            U = U + scale * np.random.randn(n_a, rank)
            V = V + scale * np.random.randn(n_b, rank)
            W = W + scale * np.random.randn(n_c, rank)

        # Adaptive sweep count: fewer for early restarts to explore more
        if restart < 10:
            sweeps = 30
        elif restart < 25:
            sweeps = 50
        else:
            sweeps = 80

        U, V, W = refine(U, V, W, sweeps, deadline)
        err = error_of(U, V, W)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())

        # If close, try polishing
        if err < 1e-3:
            U, V, W = polish(U, V, W, deadline - 3)
            err_polished = error_of(U, V, W)
            if err_polished < best_err:
                best_err = err_polished
                best = (U.copy(), V.copy(), W.copy())
            if err_polished < 1e-8:
                return U, V, W

    if best is not None:
        U, V, W = best
        # Final polish on the best solution
        if best_err < 1e-2:
            U, V, W = polish(U, V, W, deadline - 2)
            final_err = error_of(U, V, W)
            if final_err < 1e-8:
                return U, V, W
        if error_of(U, V, W) < 1e-6:
            return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under the 150s kill

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # 1) Try to find a rank-16 decomposition.
    r16 = _try_rank16(deadline)
    if r16 is not None:
        U, V, W = r16
        if _verify_decomposition(U, V, W, T, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2) Verified rank-17 (Strassen + direct).
    U, V, W = _rank17()
    if _verify_decomposition(U, V, W, T, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3) Final fallback: exact schoolbook.
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
