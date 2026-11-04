import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n


def _verify(U, V, W, r):
    """Check sum_u U[a,u] V[b,u] W[c,u] equals the multiplication tensor T[a,b,c]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = min(r, U.shape[1], V.shape[1], W.shape[1])
    # Reconstruct T[a,b,c] = sum_u U[a,u] V[b,u] W[c,u]
    # Build via einsum-free loop over u for clarity/speed balance
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for u in range(r):
        T += np.einsum('a,b,c->abc', U[:, u], V[:, u], W[:, u])
    # Target tensor: T[(i*m+j),(j*p+k),(k*n+i)] = 1, else 0
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T, target, atol=1e-9)


def _construct_strassen_54():
    """Build a rank-54 decomposition: Strassen 2x2 blocks for j-pairs (0,1),(2,3)
    across k-blocks, plus naive 2x1x2 blocks for j=4."""
    rank = 54
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        # M0: (A00+A11)(B00+B11)
        U[i0 * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col[0] += 1
        # M1: (A10+A11)B00
        c = col[0]
        U[(i0 + 1) * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] -= 1
        col[0] += 1
        # M2: A00(B01-B11)
        c = col[0]
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col[0] += 1
        # M3: A11(B10-B00)
        c = col[0]
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        col[0] += 1
        # M4: (A00+A01)B11
        c = col[0]
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        col[0] += 1
        # M5: (A10-A00)(B00+B01)
        c = col[0]
        U[(i0 + 1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col[0] += 1
        # M6: (A01-A11)(B10+B11)
        c = col[0]
        U[i0 * m + (j0 + 1), c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] -= 1
        V[(j0 + 1) * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        col[0] += 1

    def add_naive_2112(i0, j, k0):
        c = col[0]
        U[i0 * m + j, c] += 1
        V[j * p + k0, c] += 1
        W[k0 * n + i0, c] += 1
        col[0] += 1
        c = col[0]
        U[i0 * m + j, c] += 1
        V[j * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + i0, c] += 1
        col[0] += 1
        c = col[0]
        U[(i0 + 1) * m + j, c] += 1
        V[j * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        col[0] += 1
        c = col[0]
        U[(i0 + 1) * m + j, c] += 1
        V[j * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col[0] += 1

    # Strassen blocks for j-pairs (0,1) and (2,3), k-blocks (0,1),(2,3),(4,5)
    for j0 in (0, 2):
        for k0 in (0, 2, 4):
            add_strassen(0, j0, k0)
    # Naive for j=4
    for k0 in (0, 2, 4):
        add_naive_2112(0, 4, k0)

    assert col[0] == rank, f"Expected {rank} columns, got {col[0]}"
    return U, V, W


def _schoolbook():
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_merge_columns(U, V, W, r, deadline):
    """Greedy column merge: repeatedly test if removing one column can be
    compensated by adjusting the remaining columns. Return best verified result."""
    best = (U.copy(), V.copy(), W.copy(), r)
    cur_r = r
    U, V, W = U.copy(), V.copy(), W.copy()
    while cur_r > 1 and time.time() < deadline - 5.0:
        merged = False
        for drop in range(cur_r):
            if time.time() > deadline - 5.0:
                break
            # Try dropping column `drop`: see if the tensor without it can be
            # represented by a linear combination of the others.
            keep = [x for x in range(cur_r) if x != drop]
            Uk = U[:, keep]
            Vk = V[:, keep]
            Wk = W[:, keep]
            # Compute the residual tensor R = T - sum_{u in keep} U_u V_u W_u
            R = np.zeros((nm, mp, pn), dtype=np.float64)
            for u in range(len(keep)):
                R -= np.einsum('a,b,c->abc', Uk[:, u], Vk[:, u], Wk[:, u])
            # We want R = U_new ⊗ V_new ⊗ W_new (single rank-1) if a merge works.
            # Approximate: check if R is rank-1 by SVD.
            flat = R.reshape(-1, nm * pn)  # not quite; use proper unfolding
            # Unfold along the middle axis: R[a,b,c] -> matrix of shape (nm, mp*pn)
            M = R.transpose(0, 2, 1).reshape(nm, pn * mp)
            try:
                s = np.linalg.svd(M, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            if len(s) == 0:
                continue
            # Rank-1 if the second singular value is negligible
            if len(s) > 1 and s[1] > 1e-7 * max(s[0], 1e-12):
                continue
            # Rank-1: extract factors
            # Reconstruct R as u v w
            Ufull, Sfull, Vfull = np.linalg.svd(M, full_matrices=False)
            u = Ufull[:, 0] * np.sqrt(Sfull[0])
            w = Vfull[0, :] * np.sqrt(Sfull[0])
            w = w.reshape(pn, mp).T  # w[b,c]
            # Now R[a,b,c] = u[a] * w[c,b]. We need V_new such that
            # R[a,b,c] = u[a] * V_new[b] * W_new[c] with W_new[c] and V_new[b].
            # Since R = u ⊗ (w transposed as a matrix over (b,c)), factor w[b,c]
            # into V_new[b]*W_new[c]. w is itself rank-1? Check.
            Wmat = w.T  # shape (mp, pn): Wmat[b,c]
            s2 = np.linalg.svd(Wmat, compute_uv=False)
            if len(s2) > 1 and s2[1] > 1e-7 * max(s2[0], 1e-12):
                continue
            U2, S2, V2 = np.linalg.svd(Wmat, full_matrices=False)
            v = U2[:, 0] * np.sqrt(S2[0])
            wv = V2[0, :] * np.sqrt(S2[0])
            # Now R[a,b,c] = u[a] * v[b] * wv[c]
            # Add this as a new column (replacing the dropped one)
            new_U = np.concatenate([Uk, u.reshape(-1, 1)], axis=1)
            new_V = np.concatenate([Vk, v.reshape(-1, 1)], axis=1)
            new_W = np.concatenate([Wk, wv.reshape(-1, 1)], axis=1)
            # Verify
            if _verify(new_U, new_V, new_W, cur_r - 1):
                U, V, W = new_U, new_V, new_W
                cur_r -= 1
                best = (U.copy(), V.copy(), W.copy(), cur_r)
                merged = True
                break
        if not merged:
            break
    return best


def _try_svd_reduction(U, V, W, r, deadline):
    """Try to find a lower-rank exact decomposition by flattening the tensor and
    using SVD to find a low-rank structure, then factorizing back."""
    # Reconstruct the full tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for u in range(r):
        T += np.einsum('a,b,c->abc', U[:, u], V[:, u], W[:, u])
    best = (U.copy(), V.copy(), W.copy(), r)
    # Try ranks below current
    for target in range(r - 1, max(20, r - 12), -1):
        if time.time() > deadline - 5.0:
            break
        # Attempt: project the tensor onto a rank-`target` approximation and
        # check if it's exact (i.e., the tensor already has rank <= target).
        # Unfold along axis 1: matrix of shape (nm, mp*pn)
        M = T.transpose(0, 2, 1).reshape(nm, pn * mp)
        s = np.linalg.svd(M, compute_uv=False)
        # The tensor rank is at least the matrix rank. If matrix rank <= target,
        # a rank-`target` tensor decomposition may exist.
        effective_rank = int(np.sum(s > 1e-9 * max(s[0], 1e-12))) if len(s) > 0 else 0
        if effective_rank > target:
            continue
        # Try to find explicit factors: use the top-`target` singular vectors
        Ufull, Sfull, Vfull = np.linalg.svd(M, full_matrices=False)
        k = min(target, len(Sfull))
        Uf = Ufull[:, :k] * np.sqrt(Sfull[:k])
        Vf = Vfull[:k, :] * np.sqrt(Sfull[:k])
        # Vf is (k, pn*mp) -> reshape to (k, pn, mp) -> transpose to (k, mp, pn)
        Vf = Vf.reshape(k, pn, mp).transpose(0, 2, 1)  # (k, mp, pn)
        # Now T[a,b,c] = sum_l Uf[a,l] * Vf[l,b,c]. But Vf[l,b,c] is not
        # necessarily rank-1 in (b,c). Factor each slice.
        new_U = np.zeros((nm, k), dtype=np.float64)
        new_V = np.zeros((mp, k), dtype=np.float64)
        new_W = np.zeros((pn, k), dtype=np.float64)
        ok = True
        for l in range(k):
            new_U[:, l] = Uf[:, l]
            # Factor Vf[l, :, :] (shape mp x pn) into v w
            Wmat = Vf[l]
            s2 = np.linalg.svd(Wmat, compute_uv=False)
            if len(s2) > 1 and s2[1] > 1e-7 * max(s2[0], 1e-12):
                ok = False
                break
            U2, S2, V2 = np.linalg.svd(Wmat, full_matrices=False)
            new_V[:, l] = U2[:, 0] * np.sqrt(S2[0])
            new_W[:, l] = V2[0, :] * np.sqrt(S2[0])
        if not ok:
            continue
        if _verify(new_U, new_V, new_W, k):
            best = (new_U, new_V, new_W, k)
            break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with the verified rank-54 Strassen-based decomposition
    U, V, W = _construct_strassen_54()
    if _verify(U, V, W, 54):
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), 54
    else:
        # Fallback to schoolbook if Strassen construction is wrong
        U, V, W = _schoolbook()
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), 60

    # Attempt rank reduction using column-merge
    if time.time() < deadline - 15.0:
        cand = _try_merge_columns(best_U, best_V, best_W, best_rank, deadline)
        if cand[3] < best_rank:
            best_U, best_V, best_W, best_rank = cand

    # Attempt SVD-based reduction
    if time.time() < deadline - 10.0:
        cand = _try_svd_reduction(best_U, best_V, best_W, best_rank, deadline)
        if cand[3] < best_rank:
            best_U, best_V, best_W, best_rank = cand

    # Final safety: ensure the returned decomposition is valid
    if not _verify(best_U, best_V, best_W, best_rank):
        U, V, W = _schoolbook()
        best_U, best_V, best_W, best_rank = U, V, W, 60

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
