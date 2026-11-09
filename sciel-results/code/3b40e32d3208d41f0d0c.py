"""Optimized matrix multiplication decomposition for 2x5 by 5x6."""
import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact. U, V, W are numpy arrays."""
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_decomp(n, m, p, rank, deadline, seed):
    """
    Attempt to find a rank-r decomposition via numerical optimization.
    We minimize sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2.
    Use gradient-free / alternating approach with random restarts.
    """
    rng = np.random.RandomState(seed)
    best_err = None
    best_UVW = None

    U3 = np.zeros((n, m, rank))
    V3 = np.zeros((m, p, rank))
    W3 = np.zeros((p, n, rank))

    for restart in range(6):
        # random init
        U3 = rng.randn(n, m, rank) * 0.5
        V3 = rng.randn(m, p, rank) * 0.5
        W3 = rng.randn(p, n, rank) * 0.5

        # simple alternating coordinate descent on each tensor
        # gradient of L = sum_{i,j,k} (s_{ijk} - 1)^2 where s_{ijk}=sum_c U[i,j,c]V[j,k,c]W[k,i,c]
        # dL/dU[i,j,c] = 2 (s_{ijk}-1) V[j,k,c] W[k,i,c] summed over k
        # dL/dV[j,k,c] = 2 (s_{ijk}-1) U[i,j,c] W[k,i,c] summed over i
        # dL/dW[k,i,c] = 2 (s_{ijk}-1) U[i,j,c] V[j,k,c] summed over j

        def compute_s(U3, V3, W3):
            # s[i,j,k]
            s = np.zeros((n, m, p))
            for c in range(rank):
                s += U3[:, :, c] * V3[:, c, :] * W3[:, :, c].T
            return s

        for it in range(300):
            if time.time() > deadline:
                break
            s = compute_s(U3, V3, W3)
            err = np.sum((s - 1.0) ** 2)
            if best_err is None or err < best_err:
                best_err = err
                best_UVW = (U3.copy(), V3.copy(), W3.copy())
            if err < 1e-12:
                break
            lr = 0.05
            # update U3
            dU = np.zeros_like(U3)
            for c in range(rank):
                # dU[i,j,c] = 2 (s[i,j,k]-1) V[j,k,c] W[k,i,c] summed over k
                dU[:, :, c] = 2.0 * np.einsum('ijk,jkc,ki->ij', s - 1.0, V3, W3)
            U3 -= lr * dU
            # update V3
            dV = np.zeros_like(V3)
            for c in range(rank):
                dV[:, :, c] = 2.0 * np.einsum('ijk,ijc,ki->jk', s - 1.0, U3, W3)
            V3 -= lr * dV
            # update W3
            dW = np.zeros_like(W3)
            for c in range(rank):
                dW[:, :, c] = 2.0 * np.einsum('ijk,ijc,jkc->ki', s - 1.0, U3, V3)
            W3 -= lr * dW

        if best_err is not None and best_err < 1e-12:
            break

    if best_UVW is None:
        return None
    U3, V3, W3 = best_UVW
    U = U3.reshape(n*m, rank)
    V = V3.reshape(m*p, rank)
    W = W3.reshape(p*n, rank)
    if _verify(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None

def _kronecker_decomp(n, m, p):
    """
    Try to decompose into Kronecker products of smaller matrix multiplications.
    """
    best_rank = n * m * p
    best_UVW = None

    for n1 in range(1, n + 1):
        if n % n1 != 0:
            continue
        n2 = n // n1
        for m1 in range(1, m + 1):
            if m % m1 != 0:
                continue
            m2 = m // m1
            for p1 in range(1, p + 1):
                if p % p1 != 0:
                    continue
                p2 = p // p1

                rank1 = n1 * m1 * p1
                rank2 = n2 * m2 * p2
                total_rank = rank1 * rank2

                if total_rank >= best_rank:
                    continue

                U = np.zeros((n * m, total_rank))
                V = np.zeros((m * p, total_rank))
                W = np.zeros((p * n, total_rank))

                for i1 in range(n1):
                    for j1 in range(m1):
                        for k1 in range(p1):
                            for i2 in range(n2):
                                for j2 in range(m2):
                                    for k2 in range(p2):
                                        i = i1 * n2 + i2
                                        j = j1 * m2 + j2
                                        k = k1 * p2 + k2
                                        c1 = i1 * m1 * p1 + j1 * p1 + k1
                                        c2 = i2 * m2 * p2 + j2 * p2 + k2
                                        c = c1 * rank2 + c2
                                        U[i * m + j, c] = 1
                                        V[j * p + k, c] = 1
                                        W[k * n + i, c] = 1

                if _verify(U, V, W, n, m, p):
                    best_rank = total_rank
                    best_UVW = (U, V, W)

    if best_UVW is not None:
        return best_UVW
    return None

def _try_split_m(n, m, p, deadline):
    """
    Try to exploit the middle dimension m. If m = m1 + m2, we can sometimes
    combine the two partial products. Actually, matrix multiplication is linear in j,
    so splitting j just gives independent terms - no rank reduction.
    Instead, try to find a shared structure.
    """
    # This is a placeholder; the real search is in _try_find_decomp.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit

    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = (U_sb, V_sb, W_sb)
    best_rank = n * m * p

    # Try Kronecker decomposition
    kres = _kronecker_decomp(n, m, p)
    if kres is not None:
        U_k, V_k, W_k = kres
        if U_k.shape[1] < best_rank:
            best = (U_k, V_k, W_k)
            best_rank = U_k.shape[1]

    # Try numerical search for lower rank decompositions.
    # The theoretical lower bound for matrix mult rank is at least n*m*p / (something).
    # For 2x5x6, known best is around 24-30. Let's search from a few ranks.
    # We'll try ranks from a reasonable lower bound up to best_rank-1.
    # Lower bound: at least max(n*m, m*p, p*n) = max(10,30,12)=30? No, that's not a lower bound.
    # Actually a valid lower bound: rank >= n*m (since the map A -> C for fixed B is injective-ish).
    # Let's just try a range.

    random.seed(42)
    np.random.seed(42)

    # Try ranks from 20 up to best_rank-1 (descending priority: start lower)
    # But numerical search is expensive; try a handful of candidate ranks.
    candidate_ranks = []
    for r in range(20, best_rank):
        candidate_ranks.append(r)

    # Sort by rank ascending to try lowest first
    candidate_ranks.sort()

    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        res = _try_find_decomp(n, m, p, r, deadline - 5, seed=r)
        if res is not None:
            U_t, V_t, W_t = res
            if r < best_rank:
                best = (U_t, V_t, W_t)
                best_rank = r
                # Found a valid one; keep searching for even lower? 
                # But we go ascending, so lower already tried. Stop.
                break

    # Final verification of the best
    U_b, V_b, W_b = best
    if not _verify(U_b, V_b, W_b, n, m, p):
        # fallback to schoolbook
        U_b, V_b, W_b = _schoolbook(n, m, p)

    return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
