"""Numerical ALS tensor-rank search + exact half-integer recovery for 4x4*4x5 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]


def _schoolbook():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1) > 1e-7:
                    return False
    return True


def _round_half(x):
    """Round to nearest half-integer."""
    return int(round(2 * x)) / 2.0


def _round_to_half(U, V, W):
    return (np.round(2 * U) / 2, np.round(2 * V) / 2, np.round(2 * W) / 2)


def _als_decompose(target, R, n_iter=200, seed=0):
    """Alternating least squares for tensor rank-R decomposition.
    target: (n*m, m*p, p*n) tensor.
    Returns U (n*m x R), V (m*p x R), W (p*n x R) or None.
    """
    rng = np.random.default_rng(seed)
    n, m, p = DIMENSIONS
    I, J, K = n * m, m * p, p * n

    U = rng.normal(size=(I, R))
    V = rng.normal(size=(J, R))
    W = rng.normal(size=(K, R))

    for it in range(n_iter):
        # Update U: for each r, solve U[:,r] from target with V,W fixed
        for r in range(R):
            # target[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # Fix all terms except r: residual = target - sum_{r'!=r} U[i,r']V[j,r']W[k,r']
            # Then U[i,r] = <residual, V[:,r] W[:,r]> / <V[:,r]W[:,r], V[:,r]W[:,r]>
            # Vectorize: flatten target to (I, J*K)
            # Let's do it via matmul: for fixed r, define B_r[j,k] = V[j,r]*W[k,r]
            # target[i, j, k] = U[i,r]*B_r[j,k] + other
            # other = sum_{r' != r} U[i,r']*V[j,r']*W[k,r']
            # residual[i, j, k] = target[i,j,k] - other[i,j,k]
            # U[i,r] = sum_{j,k} residual[i,j,k]*B_r[j,k] / sum_{j,k} B_r[j,k]^2
            B_r = np.outer(V[:, r], W[:, r]).reshape(J, K)
            # other: (I, J, K)
            other = np.einsum('ir,jr,kr->ijk', U, V, W) - np.einsum('ir,jr,kr->ijk',
                                                                     np.where(np.arange(R) == r, 1, 0).reshape(1, 1, R) * U,
                                                                     V, W)
            # simpler: other = full - term_r
            full = np.einsum('ir,jr,kr->ijk', U, V, W)
            term_r = np.einsum('ir,jr,kr->ijk',
                               np.where(np.arange(R) == r, 1, 0).reshape(1, 1, R) * U,
                               V, W)
            residual = target - full + term_r
            num = np.einsum('ijk,jk->i', residual, B_r)
            den = np.einsum('jk,jk->', B_r, B_r)
            if den > 1e-12:
                U[:, r] = num / den

        # Update V
        for r in range(R):
            A_r = np.outer(U[:, r], W[:, r]).reshape(I, K)
            full = np.einsum('ir,jr,kr->ijk', U, V, W)
            term_r = np.einsum('ir,jr,kr->ijk',
                               np.where(np.arange(R) == r, 1, 0).reshape(1, 1, R) * U,
                               V, W)
            residual = target - full + term_r
            num = np.einsum('ijk,ik->j', residual, A_r)
            den = np.einsum('ik,ik->', A_r, A_r)
            if den > 1e-12:
                V[:, r] = num / den

        # Update W
        for r in range(R):
            A_r = np.outer(U[:, r], V[:, r]).reshape(I, J)
            full = np.einsum('ir,jr,kr->ijk', U, V, W)
            term_r = np.einsum('ir,jr,kr->ijk',
                               np.where(np.arange(R) == r, 1, 0).reshape(1, 1, R) * U,
                               V, W)
            residual = target - full + term_r
            num = np.einsum('ijk,ij->k', residual, A_r)
            den = np.einsum('ij,ij->', A_r, A_r)
            if den > 1e-12:
                W[:, r] = num / den

    return U, V, W


def _build_target():
    n, m, p = DIMENSIONS
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1
    return target


def _try_rank(R, seed, n_iter=150):
    """Try to find a rank-R decomposition via ALS, then round and verify."""
    target = _build_target()
    U, V, W = _als_decompose(target, R, n_iter=n_iter, seed=seed)
    if U is None:
        return None
    # Round to half-integers
    rU, rV, rW = _round_to_half(U, V, W)
    if _verify(rU, rV, rW):
        return {"U": rU, "V": rV, "W": rW}
    return None


def _eliminate_terms(U, V, W, max_iter=50):
    """Try to eliminate redundant terms by finding null-space directions.
    If term r can be expressed as a linear combination of others, remove it.
    We check: for each r, is the vector (U[:,r], V[:,r], W[:,r]) in the span
    of the others in the tensor product space? Hard. Instead, use a simpler
    heuristic: if two terms are identical, merge. If a term is a scalar multiple
    of another, merge.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    I, J, K = n * m, m * p, p * n

    # Build coefficient matrix: each term r contributes to entries (i,j,k)
    # coeff[i,j,k,r] = U[i,r]*V[j,r]*W[k,r]
    # We want to find if column r is in span of others.
    # This is expensive. Use a greedy approach: try removing each term and
    # check if the remaining terms can still produce the target (after
    # adjusting? No, we can't adjust). So this only works if the term is
    # exactly redundant (zero contribution) or a duplicate.

    # Simpler: merge duplicates
    seen = {}
    new_U = []
    new_V = []
    new_W = []
    for r in range(rank):
        key = (tuple(U[:, r]), tuple(V[:, r]), tuple(W[:, r]))
        if key in seen:
            # Merge: add W contribution
            idx = seen[key]
            new_W[idx] = new_W[idx] + W[:, r]
        else:
            seen[key] = len(new_U)
            new_U.append(U[:, r].copy())
            new_V.append(V[:, r].copy())
            new_W.append(W[:, r].copy())

    if len(new_U) < rank:
        nU = np.column_stack(new_U)
        nV = np.column_stack(new_V)
        nW = np.column_stack(new_W)
        if _verify(nU, nV, nW):
            return nU, nV, nW
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 140  # safety margin

    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p  # 80

    target = _build_target()

    # Try a range of ranks with multiple seeds
    for R in range(50, best_rank):
        if time.time() - start_time > deadline:
            break
        found = False
        for seed in range(5):
            if time.time() - start_time > deadline:
                break
            res = _try_rank(R, seed, n_iter=100)
            if res is not None:
                # Try elimination
                eU, eV, eW = _eliminate_terms(res["U"], res["V"], res["W"])
                rank_e = eU.shape[1]
                if rank_e < best_rank:
                    best = {"U": eU, "V": eV, "W": eW}
                    best_rank = rank_e
                found = True
                break
        if found:
            # Found a working rank, try to go lower
            continue

    # Final verification
    if not _verify(best["U"], best["V"], best["W"]):
        best = _schoolbook()

    # Convert to list-of-lists format if needed (interface expects matrices)
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
