"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Vectorize: build index arrays
    idx = np.arange(n * m * p)
    i = (idx // (m * p))
    j = ((idx // p) % m)
    k = (idx % p)
    uidx = i * m + j
    vidx = j * p + k
    widx = k * n + i
    Ue = U[uidx]  # (N, rank)
    Ve = V[vidx]  # (N, rank)
    We = W[widx]  # (N, rank)
    # sum over r of product
    s = np.einsum('ir,jr,kr->ijk', Ue, Ve, We) if False else None
    # do it via loop-free: s[n] = sum_r Ue[n,r]*Ve[n,r]*We[n,r]
    s = np.sum(Ue * Ve * We, axis=1)
    if np.max(np.abs(s - 1.0)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Strassen's 7-term decomposition for 2x2 x 2x2.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3: a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4: a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    return U, V, W

def _block_decompose(n, m, p, block_n, block_m, block_p, base_U, base_V, base_W):
    """
    Decompose n x m * m x p into blocks of size block_n x block_m * block_p.
    base_* are the per-block factor matrices.
    Returns dict or None if dimensions don't divide evenly.
    """
    if n % block_n != 0 or m % block_m != 0 or p % block_p != 0:
        return None
    if base_U.shape != (block_n * block_m, None) and base_U.shape[0] != block_n * block_m:
        pass
    nb_i = n // block_n
    nb_j = m // block_m
    nb_k = p // block_p
    brank = base_U.shape[1]
    total_rank = nb_i * nb_j * nb_k * brank

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for bi in range(nb_i):
        for bj in range(nb_j):
            for bk in range(nb_k):
                off = (bi * nb_j + bj) * nb_k * brank + bk * brank
                for r in range(brank):
                    for li in range(block_n):
                        for lj in range(block_m):
                            i = bi * block_n + li
                            j = bj * block_m + lj
                            val = base_U[li * block_m + lj, r]
                            if val != 0:
                                U[i * m + j, off + r] = val
                    for lj in range(block_m):
                        for lk in range(block_p):
                            j = bj * block_m + lj
                            k = bk * block_p + lk
                            val = base_V[lj * block_p + lk, r]
                            if val != 0:
                                V[j * p + k, off + r] = val
                    for lk in range(block_p):
                        for li in range(block_n):
                            k = bk * block_p + lk
                            i = bi * block_n + li
                            val = base_W[lk * block_n + li, r]
                            if val != 0:
                                W[k * n + i, off + r] = val
    return {"U": U, "V": V, "W": W}

def _polish_columns(U, V, W, n, m, p, tol=1e-9, max_iter=200):
    """
    Try to reduce rank by eliminating columns that are linear combinations
    of others (i.e., find a set of columns whose removal still leaves a valid
    decomposition). We do this by checking, for each column, whether it is
    redundant given the tensor constraint.
    Returns a possibly-smaller dict.
    """
    U = U.copy(); V = V.copy(); W = W.copy()
    rank = U.shape[1]
    kept = list(range(rank))
    # Precompute the tensor requirement: for each (i,j,k) we need sum over kept = 1.
    # A column c is removable if there exist coefficients a_r for r in kept\{c}
    # such that for ALL (i,j,k): sum_{r in kept\{c}} a_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    #     = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This is a linear system in a_r. We attempt to solve it.
    # Build the linear system: variables a_r (r in kept, r != c).
    # Equations indexed by (i,j,k).
    # Matrix M: rows = (i,j,k), cols = (r in kept, r != c)
    # M[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # rhs[(i,j,k)] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # If a solution exists, column c is redundant.
    # To keep it cheap, we only attempt removal when rank is moderate and we
    # use least-squares with a tight residual check.
    N = n * m * p
    # Precompute per-column outer products as vectors of length N.
    col_vec = []
    for c in range(rank):
        u = U[:, c]  # (n*m,)
        v = V[:, c]  # (m*p,)
        w = W[:, c]  # (p*n,)
        # vector over (i,j,k)
        idx = np.arange(N)
        i = idx // (m * p)
        j = (idx // p) % m
        k = idx % p
        vec = u[i * m + j] * v[j * p + k] * w[k * n + i]
        col_vec.append(vec)
    # Now attempt greedy removal.
    changed = True
    while changed and len(kept) > 0:
        changed = False
        for c in list(kept):
            if len(kept) <= 1:
                break
            others = [r for r in kept if r != c]
            M = np.column_stack([col_vec[r] for r in others])
            rhs = col_vec[c]
            try:
                sol, residuals, rank_s, sv = np.linalg.lstsq(M, rhs, rcond=None)
                if len(residuals) > 0 and residuals[0] < tol:
                    kept.remove(c)
                    changed = True
            except Exception:
                pass
    if len(kept) < rank:
        U = U[:, kept]
        V = V[:, kept]
        W = W[:, kept]
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = best["U"].shape[1]

    # Try block decomposition with Strassen 2x2 blocks.
    # For 4x4 x 4x8: blocks 2x2x2 -> 2*2*4*7 = 112
    if time.time() < deadline - 5.0:
        sU, sV, sW = _strassen_2x2()
        block_result = _block_decompose(n, m, p, 2, 2, 2, sU, sV, sW)
        if block_result is not None:
            br = block_result["U"].shape[1]
            if br < best_rank and _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
                best = block_result
                best_rank = br

    # Try block decomposition with Strassen 2x2 but only over the p dimension
    # (treat as 4x4 x 4x8 = four 4x4x4x2? not aligned). Skip.

    # Polish: attempt column elimination on the best result.
    if time.time() < deadline - 5.0:
        try:
            polished = _polish_columns(best["U"], best["V"], best["W"], n, m, p)
            pr = polished["U"].shape[1]
            if pr < best_rank and _verify(polished["U"], polished["V"], polished["W"], n, m, p):
                best = polished
                best_rank = pr
        except Exception:
            pass

    # Convert to lists of lists for the return interface.
    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
# EVOLVE-BLOCK-END
