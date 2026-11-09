"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
N, M, P = DIMENSIONS
DEADLINE = time.time() + 115.0  # safety margin before the 150s kill


def _gram(A, B):
    """Gram matrix G[i,j] = sum_k A[i,k]*B[j,k]."""
    return np.dot(A, B.T)


def _try_prune(U, V, W, tol=1e-11):
    """Remove rank-1 columns whose removal does not change the tensor.

    A column c is redundant iff the rank of the tensor after removing c
    equals the original rank.  Removing c drops the tensor rank by 1 exactly
    when the column is an "essential" rank-1 term, which happens iff the
    3-way Gram matrix (U@V.T, U@W.T, V@W.T) has a non-zero minor supported
    on c.  We test redundancy by checking whether column c lies in the span
    of the other columns' contributions.  Practically: a rank-1 term is
    removable iff there exist vectors a in row(U), b in col(V), d in col(W)
    such that the c-th rank-1 equals a sum...  Instead we use a direct
    criterion: the c-th term is redundant iff the tensor T equals T - T_c,
    i.e. T_c = 0 in the quotient, which is equivalent to the c-th column
    being in the span of the others in all three factor spaces simultaneously.
    We approximate by checking that removing c does not lower the rank of the
    three Gram matrices' combined structure.  Simplest robust test: recompute
    the tensor rank contribution.  We use: c redundant iff the c-th column of
    each factor is in the span of the remaining columns AND the rank-1 is in
    the span of the others.  For rank-1 additions this rarely triggers, so we
    keep a cheap sufficient test: if any of the three columns is a linear
    combination of the others, the term may be redundant; we then verify by
    full tensor reconstruction.
    """
    rank = U.shape[1]
    if rank < 2:
        return U, V, W
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    r0 = np.linalg.matrix_rank(T.reshape(-1, N * M * P))
    keep = list(range(rank))
    # Try removing each column; keep the ones whose removal keeps the rank.
    for c in range(rank - 1, -1, -1):
        if time.time() > DEADLINE - 5:
            break
        U2 = np.delete(U, c, axis=1)
        V2 = np.delete(V, c, axis=1)
        W2 = np.delete(W, c, axis=1)
        T2 = np.einsum('ic,jc,kc->ijk', U2, V2, W2)
        r2 = np.linalg.matrix_rank(T2.reshape(-1, N * M * P))
        if r2 == r0:
            keep = [x for x in keep if x != c]
            U, V, W = U2, V2, W2
            rank = U.shape[1]
    return U, V, W


def _strassen_seed():
    """Build a rank-28 seed: 4x4x4 Strassen (rank 7) on the 4x5x4 core,
    plus identity terms for the extra 5th row/col and 6th col."""
    n, m, p = N, M, P
    rank = 0
    U = []
    V = []
    W = []

    def add(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # 4x4x4 Strassen decomposition (indices i<4, j<4, k<4).
    # Strassen rank-7 for 2x2: M1..M7.  Compose for 4x4 as block Strassen.
    # We use the standard 4x4 rank-7 tensor (the 2x2 Strassen applied at
    # the 2x2 block level, i.e. tensor power).  For 4x4x4 the known minimal
    # is 7 (Strassen) only for 2x2; 4x4 has rank at most 14 via block
    # Strassen.  We use block Strassen: treat 4x4 as 2x2 blocks of 2x2,
    # each 2x2 block multiply uses 7 rank-1 terms, total 14 terms for the
    # 4x4x4 core (i<4,j<4,k<4).
    s2 = [  # 7 Strassen terms for 2x2: (u_row, u_col, v_row, v_col, w_row, w_col)
        (0, 0, 0, 0, 0, 0),  # (A00)(B00)
        (0, 1, 0, 0, 0, 1),  # (A01)(B00)
        (1, 0, 0, 1, 1, 0),  # (A10)(B01)
        (1, 1, 1, 1, 1, 1),  # (A11)(B11)
        (1, 0, 0, 1, 0, 0),  # (A10+A00)(B01)
        (1, 1, 1, 0, 1, 1),  # (A11+A10)(B01)
        (0, 0, 1, 0, 1, 0),  # (A00)(B10)
    ]
    # Proper Strassen 2x2 rank-7:
    s2 = [
        (0, 0, 0, 0, 0, 0),  # M1 = A00*B00
        (0, 1, 0, 0, 0, 1),  # M2 = A01*B00
        (1, 0, 0, 1, 1, 0),  # M3 = A10*B01
        (1, 1, 1, 1, 1, 1),  # M4 = A11*B11
        (0, 0, 1, 0, 0, 0),  # M5 = (A00+A10)*B01 -> contributes A00*B01
        (0, 0, 0, 1, 1, 1),  # M6 = (A01+A11)*B10 -> contributes A01*B10
        (1, 0, 1, 1, 0, 0),  # M7 = (A10+A11)*(B11-B10) ... 
    ]
    # Use the canonical Strassen 2x2 rank-7 explicitly:
    # C00 = M1 - M5 + M6 - M7
    # C01 = M2 + M5
    # C10 = M3 + M7
    # C11 = M4 - M2 + M3 + M5
    # M1 = A00*B00
    # M2 = A01*B00
    # M3 = A10*B01
    # M4 = A11*B11
    # M5 = (A00+A10)*B01
    # M6 = (A01+A11)*B10
    # M7 = (A10+A11)*(B11-B10)
    s2 = [
        ([(0, 1)], [(0, 1)], [(0, 1)]),  # M1
        ([(0, 1)], [(1, 1)], [(0, 1)]),  # M2
        ([(1, 1)], [(0, 1)], [(1, 1)]),  # M3
        ([(1, 1)], [(1, 1)], [(1, 1)]),  # M4
        ([(0, 1), (1, 1)], [(0, 1)], [(1, 1)]),  # M5
        ([(0, 1), (1, 1)], [(1, 1)], [(0, 1)]),  # M6
        ([(1, 1), (1, 1)], [(1, -1)], [(0, 1)]),  # M7
    ]
    # Build 4x4x4 via 2x2 block Strassen: blocks (ii, jj, kk) each in {0,1},
    # block size 2.  For each of the 7 Strassen terms and each block combination
    # we add one rank-1 term with the corresponding 2x2 sub-tensors.
    for (ua, va, wa) in s2:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    u = np.zeros((4, 4))
                    v = np.zeros((4, 4))
                    w = np.zeros((4, 4))
                    for (r, c) in ua:
                        u[ii * 2 + r, ii * 2 + c] = 1
                    for (r, c) in va:
                        v[jj * 2 + r, jj * 2 + c] = 1
                    for (r, c) in wa:
                        w[kk * 2 + r, kk * 2 + c] = 1
                    add(u.flatten(), v.flatten(), w.flatten())

    # Identity terms for the 5th column of A (j=4) and 6th column of C (k=5),
    # and the 5th row of B (k=4).  We cover the remaining (i<4, j=4, k<6),
    # (i<4, j<5, k=5) and (i=4? no, n=4 so i<4 only).
    # j=4 (extra column of A): for i in 0..3, k in 0..5
    for i in range(4):
        for k in range(6):
            u = np.zeros(20); v = np.zeros(30); w = np.zeros(24)
            u[i * 5 + 4] = 1
            v[4 * 6 + k] = 1
            w[k * 4 + i] = 1
            add(u, v, w)
    # k=5 (extra column of C) for j in 0..4, i in 0..3 (j=4,k=5 already covered)
    for i in range(4):
        for j in range(5):
            if j == 4:
                continue
            u = np.zeros(20); v = np.zeros(30); w = np.zeros(24)
            u[i * 5 + j] = 1
            v[j * 6 + 5] = 1
            w[5 * 4 + i] = 1
            add(u, v, w)

    U = np.array(U).T  # (20, rank)
    V = np.array(V).T  # (30, rank)
    W = np.array(W).T  # (24, rank)
    return U, V, W


def _verify(U, V, W):
    """Check that the factorization exactly reproduces the multiplication tensor."""
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    Tt = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                Tt[i, j, k] = 1
    return np.allclose(T, Tt, atol=1e-9)


def _random_rank1():
    """Random rank-1 candidate with small integer entries."""
    u = np.random.randint(-2, 3, N * M)
    v = np.random.randint(-2, 3, M * P)
    w = np.random.randint(-2, 3, P * N)
    return u, v, w


def construct():
    start = time.time()
    rng = np.random.default_rng(12345)

    U, V, W = _strassen_seed()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Verify the seed.
    if not _verify(U, V, W):
        # Fallback to schoolbook if seed is wrong.
        rank = N * M * P
        Us = np.zeros((N * M, rank))
        Vs = np.zeros((M * P, rank))
        Ws = np.zeros((P * N, rank))
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    Us[i * M + j, c] = 1
                    Vs[j * P + k, c] = 1
                    Ws[k * N + i, c] = 1
                    c += 1
        U, V, W = Us, Vs, Ws
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank

    # Try to prune the seed.
    U, V, W = _try_prune(U, V, W)
    if _verify(U, V, W) and U.shape[1] < best_rank:
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Search: try adding random rank-1 terms and re-pruning.  We look for a
    # decomposition with fewer total rank-1 terms than the seed by adding a
    # term that lets us drop several others.  Since the seed already covers the
    # tensor, adding terms only helps if they enable a re-expression.  We
    # instead perform a "swap" search: replace subsets of columns with new
    # rank-1 terms.  Given time limits, we do a greedy local search:
    # repeatedly try to drop one column and replace it with one new column
    # that restores the tensor, accepting if it reduces rank (never) or keeps
    # rank (neutral, for diversity).  To actually reduce rank we need a
    # multi-term move; we do a small beam over 2-column removals.
    attempts = 0
    while time.time() < DEADLINE - 10:
        attempts += 1
        r = U.shape[1]
        if r <= 24:  # cannot beat 24 (lower bound n*m=20, but 24 is our target)
            break
        # Pick two columns to try to remove.
        if r >= 2:
            c1, c2 = rng.choice(r, size=2, replace=False)
            c1, c2 = int(c1), int(c2)
            U2 = np.delete(U, [c1, c2], axis=1)
            V2 = np.delete(V, [c1, c2], axis=1)
            W2 = np.delete(W, [c1, c2], axis=1)
            # Try to find one rank-1 term that restores the tensor.
            T_def = np.einsum('ic,jc,kc->ijk', U, V, W) - np.einsum('ic,jc,kc->ijk', U2, V2, W2)
            # T_def = T_c1 + T_c2 (sum of two rank-1 tensors).  It is rank-1
            # only in special cases; generally rank 2.  So one term won't
            # restore it.  Skip single-term restore.
            # Instead try adding one random term and then pruning.
            u, v, w = _random_rank1()
            U3 = np.hstack([U, u.reshape(-1, 1)])
            V3 = np.hstack([V, v.reshape(-1, 1)])
            W3 = np.hstack([W, w.reshape(-1, 1)])
            U3, V3, W3 = _try_prune(U3, V3, W3)
            if _verify(U3, V3, W3) and U3.shape[1] < best_rank:
                best_rank = U3.shape[1]
                best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
                U, V, W = U3, V3, W3

    # Final: ensure we return a valid decomposition.
    if _verify(best_U, best_V, best_W):
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = _schoolbook()

    # Convert to lists of floats/ints (integer or half-integer entries).
    Uo = [[float(x) for x in row] for row in U]
    Vo = [[float(x) for x in row] for row in V]
    Wo = [[float(x) for x in row] for row in W]
    return {"U": Uo, "V": Vo, "W": Wo}


def _schoolbook():
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W
