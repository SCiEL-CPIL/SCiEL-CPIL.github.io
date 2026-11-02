"""Tensor slice sparsification approach for 4x4 * 4x5 matrix multiplication.
Structural change: joint coordinate descent with term pruning, exploiting
the 5-slice structure of the tensor."""
import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 5]


def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=atol)


def _round_to_half_integer(x):
    return round(x * 2) / 2.0


def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])


def _build_kronecker_baseline(n=4, m=4, p=5):
    """Build a structured baseline using Strassen for 2x2x2 and naive for strips.
    Returns U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank)."""
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def ui(gi, gj):
        return gi * m + gj

    def vi(gj, gk):
        return gj * p + gk

    def wi(gk, gi):
        return gk * n + gi

    def strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[ui(ai, aj)] = 1; u[ui(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[vi(bj, bk)] = 1; v[vi(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[wi(bk, ai)] = 1; w[wi(bk+1, ai+1)] = 1
        add_term(u, v, w)
        # M2 = (c+d)e
        u = np.zeros(n*m); u[ui(ai+1, aj)] = 1; u[ui(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[vi(bj, bk)] = 1
        w = np.zeros(p*n); w[wi(bk, ai+1)] = 1; w[wi(bk+1, ai+1)] = -1
        add_term(u, v, w)
        # M3 = a(f-h)
        u = np.zeros(n*m); u[ui(ai, aj)] = 1
        v = np.zeros(m*p); v[vi(bj, bk+1)] = 1; v[vi(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[wi(bk+1, ai)] = 1; w[wi(bk+1, ai+1)] = 1
        add_term(u, v, w)
        # M4 = d(g-e)
        u = np.zeros(n*m); u[ui(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[vi(bj+1, bk)] = 1; v[vi(bj, bk)] = -1
        w = np.zeros(p*n); w[wi(bk, ai)] = 1; w[wi(bk, ai+1)] = 1
        add_term(u, v, w)
        # M5 = (a+b)h
        u = np.zeros(n*m); u[ui(ai, aj)] = 1; u[ui(ai, aj+1)] = 1
        v = np.zeros(m*p); v[vi(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[wi(bk, ai)] = -1; w[wi(bk+1, ai)] = 1
        add_term(u, v, w)
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[ui(ai+1, aj)] = 1; u[ui(ai, aj)] = -1
        v = np.zeros(m*p); v[vi(bj, bk)] = 1; v[vi(bj, bk+1)] = 1
        w = np.zeros(p*n); w[wi(bk+1, ai+1)] = 1
        add_term(u, v, w)
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[ui(ai, aj+1)] = 1; u[ui(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[vi(bj+1, bk)] = 1; v[vi(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[wi(bk, ai)] = 1
        add_term(u, v, w)

    def block_2x2x3(ai, aj, bj, bk):
        """2x2x3: Strassen for first 2 cols + 4 naive for 3rd col"""
        strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[ui(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[vi(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[wi(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # 4x4x5 = 2x(2x4x5) = 2x(2x(2x2x2 + 2x2x3))
    # C00 (2x2) = A00*B00 + A01*B10
    strassen_2x2_2(0, 0, 0, 0)
    strassen_2x2_2(0, 2, 2, 0)
    # C10 (2x2) = A10*B00 + A11*B10
    strassen_2x2_2(2, 0, 0, 0)
    strassen_2x2_2(2, 2, 2, 0)
    # C01 (2x3) = A00*B01 + A01*B11
    block_2x2x3(0, 0, 0, 2)
    block_2x2x3(0, 2, 2, 2)
    # C11 (2x3) = A10*B01 + A11*B11
    block_2x2x3(2, 0, 0, 2)
    block_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat


def _prune_terms(U, V, W, n, m, p, threshold=1e-6, max_prunes=5):
    """Remove terms where the W-column is nearly zero or where the term contributes
    negligibly to the tensor. Returns (U, V, W, n_pruned) or None."""
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    rank = U.shape[1]

    # Compute contribution of each term to the tensor
    # Term r contributes: Ur[:,:,r] outer Vr[:,:,r] outer Wr[:,:,r] (elementwise)
    # We measure the L2 norm of each term's contribution
    keep = np.ones(rank, dtype=bool)
    pruned = 0

    for _ in range(max_prunes):
        # Find the term with smallest contribution
        norms = np.zeros(rank)
        for r in range(rank):
            contrib = Ur[:, :, r] * Vr[:, :, r] * Wr[:, :, r]
            norms[r] = np.linalg.norm(contrib)

        # Find smallest norm among kept terms
        min_idx = np.argmin(np.where(keep, norms, np.inf))
        if norms[min_idx] > threshold:
            break

        # Try removing this term
        keep[min_idx] = False
        pruned += 1

        # Rebuild and verify
        U_t = U[:, keep]
        V_t = V[:, keep]
        W_t = W[:, keep]
        if _verify_fast(U_t.tolist(), V_t.tolist(), W_t.tolist(), n, m, p, atol=1e-9):
            pass
        else:
            keep[min_idx] = True
            pruned -= 1
            break

    if pruned == 0:
        return None

    U_t = U[:, keep]
    V_t = V[:, keep]
    W_t = W[:, keep]
    if _verify_fast(U_t.tolist(), V_t.tolist(), W_t.tolist(), n, m, p, atol=1e-9):
        return U_t, V_t, W_t, pruned
    return None


def _coordinate_descent(U, V, W, n, m, p, max_iters=100, deadline=None, tol=1e-14):
    """Coordinate descent: optimize each term independently.
    For term r, with other terms fixed, solve for U[:,r], V[:,r], W[:,r] to best
    approximate the residual tensor."""
    Ur = U.copy()
    Vr = V.copy()
    Wr = W.copy()
    rank = Ur.shape[1]

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Compute current tensor
        T_cur = np.einsum('ijr,jkr,kir->ijk', Ur.reshape(n,m,rank),
                           Vr.reshape(m,p,rank), Wr.reshape(p,n,rank))
        residual = 1.0 - T_cur

        max_change = 0.0

        for r in range(rank):
            if deadline is not None and time.time() > deadline:
                break

            # Remove term r's contribution
            T_without_r = T_cur - Ur.reshape(n,m,rank)[:,:,r] * Vr.reshape(m,p,rank)[:,:,r] * Wr.reshape(p,n,rank)[:,:,r]
            # Actually, let's compute residual for term r:
            # We want: sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r'] = 1
            # Fix all r' != r, solve for U[:,r], V[:,r], W[:,r]
            # This is a 3-way tensor problem - use alternating optimization

            # Optimize U[:,r]: for each (i,j), U[i*m+j, r] * sum_k V[j*p+k,r] * W[k*n+i,r] should contribute
            # Let's use lstsq approach
            # For U: fix V[:,r] and W[:,r], optimize U[:,r]
            # For each (i,j): U[i*m+j, r] * (V[:,r] * W[:,r] projected) = residual contribution
            # This is complex; use simple gradient-free update

            # Simpler: use the fact that if we fix V[:,r] and W[:,r],
            # then for each (i,j), U[i*m+j,r] should satisfy:
            # U[i*m+j,r] * sum_k V[j*p+k,r] W[k*n+i,r] = residual[i,j,?]
            # But this couples across k. Let's use a different approach.

            # Use rank-1 approximation of residual via SVD-like approach
            # Residual is 3D; we want to find u,v,w such that u⊗v⊗w ≈ residual
            # Use iterative: fix v,w -> solve u; fix u,w -> solve v; fix u,v -> solve w

            u_old = Ur[:, r].copy()
            v_old = Vr[:, r].copy()
            w_old = Wr[:, r].copy()

            for inner in range(3):
                # Solve for u given v, w
                # u[i*m+j] * sum_k v[j*p+k] * w[k*n+i] ≈ residual[i,j,*]
                # For each (i,j), the coefficient is: c[i,j] = sum_k v[j*p+k] * w[k*n+i]
                # Then u[i*m+j] * c[i,j] should approximate the "average" residual
                # Better: minimize sum_{i,j,k} (residual[i,j,k] - u[i,j]*v[j,k]*w[k,i])^2 over u
                # = sum_{i,j} (sum_k residual[i,j,k]*v[j,k]*w[k,i] - u[i,j]*sum_k v[j,k]*w[k,i])^2
                # Optimal: u[i,j] = sum_k residual[i,j,k]*v[j,k]*w[k,i] / sum_k v[j,k]*w[k,i]

                v_w = Vr.reshape(m,p,1) * Wr.reshape(p,n,1)  # (m,p,1)*(p,n,1) - need alignment
                # v[j,k] * w[k,i] for fixed j,i: sum over k
                # Let's compute properly
                vw = np.einsum('jk,ki->ji', Vr.reshape(m,p,1).squeeze(), Wr.reshape(p,n,1).squeeze())
                # vw[j,i] = sum_k V[j*p+k, r] * W[k*n+i, r]

                rw = np.einsum('ijk,jk,ki->ij', residual,
                               Vr.reshape(m,p,1).squeeze(),
                               Wr.reshape(p,n,1).squeeze())
                # rw[i,j] = sum_k residual[i,j,k] * V[j*p+k,r] * W[k*n+i,r]

                denom = vw.T  # denom[i,j] = sum_k V[j*p+k,r] * W[k*n+i,r]
                with np.errstate(divide='ignore', invalid='ignore'):
                    u_new = np.where(np.abs(denom) > 1e-12, rw / denom, 0.0)

                Ur[:, r] = u_new.flatten()

                # Solve for v given u, w
                uw = np.einsum('ij,ki->jk', Ur.reshape(n,m,1).squeeze(),
                               Wr.reshape(p,n,1).squeeze())
                # uw[j,k] = sum_i U[i*m+j,r] * W[k*n+i,r]

                ruw = np.einsum('ijk,ij,ki->jk', residual,
                                Ur.reshape(n,m,1).squeeze(),
                                Wr.reshape(p,n,1).squeeze())
                # ruw[j,k] = sum_i residual[i,j,k] * U[i*m+j,r] * W[k*n+i,r]

                denom_v = uw
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_new = np.where(np.abs(denom_v) > 1e-12, ruw / denom_v, 0.0)

                Vr[:, r] = v_new.flatten()

                # Solve for w given u, v
                uv = np.einsum('ij,jk->ik', Ur.reshape(n,m,1).squeeze(),
                               Vr.reshape(m,p,1).squeeze())
                # uv[i,k] = sum_j U[i*m+j,r] * V[j*p+k,r]

                ruv = np.einsum('ijk,ij,jk->ki', residual,
                                Ur.reshape(n,m,1).squeeze(),
                                Vr.reshape(m,p,1).squeeze())
                # ruv[k,i] = sum_j residual[i,j,k] * U[i*m+j,r] * V[j*p+k,r]

                denom_w = uv.T  # denom_w[k,i] = sum_j U[i*m+j,r] * V[j*p+k,r]
                with np.errstate(divide='ignore', invalid='ignore'):
                    w_new = np.where(np.abs(denom_w) > 1e-12, ruv / denom_w, 0.0)

                Wr[:, r] = w_new.flatten()

            max_change = max(max_change,
                            np.linalg.norm(Ur[:, r] - u_old),
                            np.linalg.norm(Vr[:, r] - v_old),
                            np.linalg.norm(Wr[:, r] - w_old))

        if max_change < tol:
            break

    return Ur, Vr, Wr


def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline, seed=42):
    """Try to find a rank-target_rank decomposition starting from given U,V,W."""
    np.random.seed(seed)
    rank = U.shape[1]

    if target_rank >= rank:
        # Just try coordinate descent on the existing solution
        U_new, V_new, W_new = _coordinate_descent(U, V, W, n, m, p,
                                                   max_iters=50, deadline=deadline)
        U_r = _round_matrix_to_half_int(U_new)
        V_r = _round_matrix_to_half_int(V_new)
        W_r = _round_matrix_to_half_int(W_new)
        if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Initialize: take first target_rank columns + random perturbation
    U_init = U[:, :target_rank].copy()
    V_init = V[:, :target_rank].copy()
    W_init = W[:, :target_rank].copy()

    # Add small random perturbation to break symmetry
    U_init += np.random.randn(n*m, target_rank) * 0.01
    V_init += np.random.randn(m*p, target_rank) * 0.01
    W_init += np.random.randn(p*n, target_rank) * 0.01

    # Coordinate descent
    U_new, V_new, W_new = _coordinate_descent(U_init, V_init, W_init, n, m, p,
                                               max_iters=200, deadline=deadline)

    # Round and verify
    U_r = _round_matrix_to_half_int(U_new)
    V_r = _round_matrix_to_half_int(V_new)
    W_r = _round_matrix_to_half_int(W_new)

    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r, V_r, W_r

    # Try without rounding - check if close enough
    if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
        return U_new, V_new, W_new

    return None


def _merge_identical(U, V, W, n, m, p):
    """Merge terms with identical (U_col, V_col) patterns."""
    groups = defaultdict(list)
    for r in range(U.shape[1]):
        key_u = tuple(U[:, r])
        key_v = tuple(V[:, r])
        groups[(key_u, key_v)].append(r)

    new_U = []
    new_V = []
    new_W = []

    for (ku, kv), idxs in groups.items():
        u_col = U[:, idxs[0]]
        v_col = V[:, idxs[0]]
        if len(idxs) == 1:
            w_col = W[:, idxs[0]]
        else:
            w_col = np.zeros_like(W[:, 0])
            for idx in idxs:
                w_col += W[:, idx]
        new_U.append(u_col)
        new_V.append(v_col)
        new_W.append(w_col)

    if len(new_U) >= U.shape[1]:
        return None

    U_new = np.column_stack(new_U)
    V_new = np.column_stack(new_V)
    W_new = np.column_stack(new_W)

    if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
        return U_new, V_new, W_new
    return None


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS

    # Build baseline
    U, V, W = _build_kronecker_baseline(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Verify baseline
    if not _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return _schoolbook()

    # Step 1: Merge identical terms
    if time.time() < deadline - 100:
        merged = _merge_identical(best_U, best_V, best_W, n, m, p)
        if merged is not None:
            best_U, best_V, best_W = merged
            best_rank = best_U.shape[1]

    # Step 2: Prune negligible terms
    if time.time() < deadline - 80:
        pruned = _prune_terms(best_U, best_V, best_W, n, m, p, threshold=1e-6, max_prunes=10)
        if pruned is not None:
            U_p, V_p, W_p, n_pruned = pruned
            if U_p.shape[1] < best_rank:
                best_U, best_V, best_W = U_p, V_p, W_p
                best_rank = best_U.shape[1]

    # Step 3: Coordinate descent to refine and potentially enable further pruning
    if time.time() < deadline - 60:
        U_cd, V_cd, W_cd = _coordinate_descent(best_U, best_V, best_W, n, m, p,
                                                max_iters=100, deadline=deadline - 60)
        U_r = _round_matrix_to_half_int(U_cd)
        V_r = _round_matrix_to_half_int(V_cd)
        W_r = _round_matrix_to_half_int(W_cd)
        if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            best_U, best_V, best_W = U_r, V_r, W_r

            # Try pruning again after coordinate descent
            if time.time() < deadline - 40:
                pruned2 = _prune_terms(best_U, best_V, best_W, n, m, p, threshold=1e-6, max_prunes=10)
                if pruned2 is not None:
                    U_p2, V_p2, W_p2, _ = pruned2
                    if U_p2.shape[1] < best_rank:
                        best_U, best_V, best_W = U_p2, V_p2, W_p2
                        best_rank = best_U.shape[1]

    # Step 4: Try reducing rank further
    for target in range(best_rank - 1, max(30, best_rank - 20), -1):
        if time.time() > deadline - 20:
            break

        found = False
        for seed in range(8):
            if time.time() > deadline - 20:
                break
            result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, target,
                                       deadline - 20, seed=seed)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target
                found = True
                break

        if found:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
