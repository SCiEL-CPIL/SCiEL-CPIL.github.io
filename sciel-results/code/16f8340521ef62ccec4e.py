"""Search for a rank-minimal exact decomposition of the 5x5x5 matrix multiplication tensor.

Strategy:
  1. The matmul tensor T[i,j,k] = 1 if i==k (for n=m=p=5) is invariant under permuting
     the three modes. Any rank-r decomposition can be symmetrized to U=V=W.
  2. Search for a symmetric decomposition U=V=W with entries in {+1,-1} (or small
     integers) such that sum_c U[i,c]*U[j,c]*U[k,c] == T[i,j,k].
  3. Use time-budgeted random local search (flip a random sign) to minimize the
     mismatch, with a final exact-verification pass.
  4. If no sub-125 symmetric ±1 decomposition is found, fall back to the schoolbook
     rank-125 decomposition (guaranteed valid).

The output follows the trace convention W[k*n+i, r].
"""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]
N = 5
TIME_LIMIT = 140.0  # safety margin below the 150s wall-clock kill


def _schoolbook():
    """Return the guaranteed-valid schoolbook rank-125 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _build_target():
    """T[i,j,k] = 1 if i==k else 0, for n=m=p=5. Shape (5,5,5)."""
    T = np.zeros((N, N, N), dtype=np.int64)
    for i in range(N):
        for k in range(N):
            if i == k:
                for j in range(N):
                    T[i, j, k] = 1
    return T


def _symmetric_decompose(U):
    """Given a 2D array U of shape (N, rank), return the dict with U=V=W=U."""
    U_list = [[int(x) for x in row] for row in U]
    return {"U": U_list, "V": U_list, "W": U_list}


def _verify(U, V, W, T):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == T[i,j,k] for all i,j,k."""
    n, m, p = DIMENSIONS
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    # For symmetric case U=V=W of shape (N, rank)
    # U[i*m+j, c] = Ua[i*N+j, c], V[j*p+k,c]=Va[j*N+k,c], W[k*n+i,c]=Wa[k*N+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.dot(Ua[i * m + j], Va[j * p + k]) if False else
                          np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i]))
                if val != T[i, j, k]:
                    return False
    return True


def _mismatch(U, T):
    """Count how many (i,j,k) entries of the reconstructed tensor differ from T."""
    n = N
    # U is (N, rank); reconstruction R[i,j,k] = sum_c U[i,c]*U[j,c]*U[k,c]
    R = np.einsum('ic,jc,kc->ijk', U, U, U)
    return int(np.sum(R != T))


def _search_symmetric(T, deadline):
    """Random local search for a symmetric ±1 decomposition U=V=W with U of shape (N, rank).

    Returns (U, mismatch) for the best found, or None if no improvement over schoolbook.
    """
    n = N
    best_mis = n * n * n  # 125
    best_U = None

    # Start with random ±1 matrices of decreasing rank.
    # Try ranks: we want rank < 125. Start around 50-80 and go lower.
    for rank in [64, 56, 48, 40, 32, 24, 16, 12, 8, 6, 4, 2]:
        if time.time() > deadline:
            break
        # Several random restarts per rank
        for restart in range(200):
            if time.time() > deadline:
                break
            U = np.random.choice([-1, 1], size=(n, rank))
            mis = _mismatch(U, T)
            if mis < best_mis:
                best_mis = mis
                best_U = U.copy()
            if mis == 0:
                return U, 0

            # Local search: flip random signs to reduce mismatch
            cur_U = U
            cur_mis = mis
            stuck = 0
            while time.time() < deadline and stuck < 50:
                # pick a random row and column to flip
                r = random.randrange(n)
                c = random.randrange(rank)
                cur_U[r, c] *= -1
                new_mis = _mismatch(cur_U, T)
                if new_mis < cur_mis:
                    cur_mis = new_mis
                    stuck = 0
                    if cur_mis < best_mis:
                        best_mis = cur_mis
                        best_U = cur_U.copy()
                    if cur_mis == 0:
                        return cur_U, 0
                else:
                    cur_U[r, c] *= -1  # revert
                    stuck += 1
    return best_U, best_mis


def _search_general(T, deadline):
    """Search for a general (non-symmetric) ±1 decomposition U,V,W with rank < 125.

    Uses random initialization + local search on the full triple.
    Returns (U,V,W,mismatch) or None.
    """
    n = N
    best_mis = n * n * n
    best = None

    for rank in [40, 32, 24, 16, 12, 8, 6]:
        if time.time() > deadline:
            break
        for restart in range(50):
            if time.time() > deadline:
                break
            U = np.random.choice([-1, 1], size=(n * n, rank))
            V = np.random.choice([-1, 1], size=(n * n, rank))
            W = np.random.choice([-1, 1], size=(n * n, rank))

            def mis_of(U, V, W):
                # R[i,j,k] = sum_c U[i*n+j,c]*V[j*n+k,c]*W[k*n+i,c]
                cnt = 0
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            val = int(np.sum(U[i * n + j] * V[j * n + k] * W[k * n + i]))
                            if val != T[i, j, k]:
                                cnt += 1
                return cnt

            cur_mis = mis_of(U, V, W)
            if cur_mis < best_mis:
                best_mis = cur_mis
                best = (U.copy(), V.copy(), W.copy())
            if cur_mis == 0:
                return U, V, W, 0

            # Local search
            stuck = 0
            while time.time() < deadline and stuck < 30:
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    idx = random.randrange(n * n)
                    c = random.randrange(rank)
                    U[idx, c] *= -1
                elif mat == 1:
                    idx = random.randrange(n * n)
                    c = random.randrange(rank)
                    V[idx, c] *= -1
                else:
                    idx = random.randrange(n * n)
                    c = random.randrange(rank)
                    W[idx, c] *= -1

                new_mis = mis_of(U, V, W)
                if new_mis < cur_mis:
                    cur_mis = new_mis
                    stuck = 0
                    if cur_mis < best_mis:
                        best_mis = cur_mis
                        best = (U.copy(), V.copy(), W.copy())
                    if cur_mis == 0:
                        return U, V, W, 0
                else:
                    # revert
                    if mat == 0:
                        U[idx, c] *= -1
                    elif mat == 1:
                        V[idx, c] *= -1
                    else:
                        W[idx, c] *= -1
                    stuck += 1
    return best[0], best[1], best[2], best_mis if best else None


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    T = _build_target()

    # First, try symmetric ±1 search (fast, exploits symmetry).
    best_U, best_mis = _search_symmetric(T, deadline)
    if best_U is not None and best_mis == 0:
        # Verify exactly
        d = _symmetric_decompose(best_U)
        if _verify(d["U"], d["V"], d["W"], T):
            return d

    # Try general ±1 search if time remains.
    if time.time() < deadline - 5.0:
        res = _search_general(T, deadline)
        if res is not None:
            U, V, W, mis = res
            if mis == 0:
                d = {
                    "U": [[int(x) for x in row] for row in U],
                    "V": [[int(x) for x in row] for row in V],
                    "W": [[int(x) for x in row] for row in W],
                }
                if _verify(d["U"], d["V"], d["W"], T):
                    return d

    # Fallback: guaranteed-valid schoolbook rank-125.
    return _schoolbook()
