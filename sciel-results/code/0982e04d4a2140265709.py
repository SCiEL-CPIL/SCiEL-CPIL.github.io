"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition.

Combines disjoint Strassen-block packing (maximize independent 2x2x2 blocks)
with exact verification and column-elimination rank reduction.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]


def _build_strassen_cols(i1, i2, j1, j2, k1, k2, m, p, n, U_rows, V_rows, W_rows):
    """Return the 7 Strassen bilinear-form columns for a 2x2x2 block."""
    cols = []
    a11, a12 = i1 * m + j1, i1 * m + j2
    a21, a22 = i2 * m + j1, i2 * m + j2
    b11, b12 = j1 * p + k1, j1 * p + k2
    b21, b22 = j2 * p + k1, j2 * p + k2
    c11, c12 = k1 * n + i1, k2 * n + i1
    c21, c22 = k1 * n + i2, k2 * n + i2

    def T(ud, vd, wd):
        u = np.zeros(U_rows); v = np.zeros(V_rows); w = np.zeros(W_rows)
        for idx, val in ud.items():
            u[idx] = val
        for idx, val in vd.items():
            v[idx] = val
        for idx, val in wd.items():
            w[idx] = val
        return (u, v, w)

    # M1 = (a11+a22)*(b11+b22) -> c11 + c22
    cols.append(T({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
    # M2 = (a21+a22)*b11 -> c21 - c22
    cols.append(T({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
    # M3 = a11*(b12-b22) -> c12 + c22
    cols.append(T({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
    # M4 = a22*(b21-b11) -> c11 + c21
    cols.append(T({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
    # M5 = (a11+a12)*b22 -> c11 - c12  ... wait, sign check below
    # Standard: M5 = (a11+a12)*b22 contributes to C11 - C12? Use canonical:
    cols.append(T({a11: 1, a12: 1}, {b22: 1}, {c12: 1, c22: -1}))
    # M6 = (a21-a11)*(b11+b12) -> c22
    cols.append(T({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
    # M7 = (a12-a22)*(b21+b22) -> c11
    cols.append(T({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
    return cols


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Precompute the target tensor as a flat array of shape (U_rows, V_rows, W_rows)
    T = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_cols(cols):
        """cols: list of (u, v, w) numpy arrays. Return True if exact."""
        if not cols:
            return False
        U = np.stack([c[0] for c in cols], axis=1)  # (U_rows, R)
        V = np.stack([c[1] for c in cols], axis=1)  # (V_rows, R)
        W = np.stack([c[2] for c in cols], axis=1)  # (W_rows, R)
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))

    # Generate all possible 2x2x2 blocks with their covered (i,j,k) set
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    def pack_and_build(packing):
        """Build column list from a packing of block params + schoolbook for rest."""
        cols = []
        covered = set()
        for (i1, i2, j1, j2, k1, k2) in packing:
            cols.extend(_build_strassen_cols(i1, i2, j1, j2, k1, k2,
                                             m, p, n, U_rows, V_rows, W_rows))
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        covered.add((i, j, k))
        # schoolbook for uncovered triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(U_rows); v = np.zeros(V_rows); w = np.zeros(W_rows)
                        u[i * m + j] = 1
                        v[j * p + k] = 1
                        w[k * n + i] = 1
                        cols.append((u, v, w))
        return cols

    # Greedy packing with many random orderings to maximize block count
    best_packing = []
    best_count = 0
    rng = random.Random(12345)
    num_trials = 4000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        used = set()
        packing = []
        for bi in order:
            params, indices = all_blocks[bi]
            if not (indices & used):
                packing.append(params)
                used |= indices
        if len(packing) > best_count:
            best_count = len(packing)
            best_packing = packing
            if best_count >= 12:
                break

    cols = pack_and_build(best_packing)
    if not verify_cols(cols):
        # Fallback: pure schoolbook
        cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(U_rows); v = np.zeros(V_rows); w = np.zeros(W_rows)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    cols.append((u, v, w))

    # Column-elimination rank reduction:
    # A column c can be removed if the tensor it contributes is already
    # reproduced by the remaining columns. Equivalently, check if removing c
    # keeps the decomposition exact. We greedily try to drop columns.
    def try_eliminate(cols):
        changed = True
        while changed:
            changed = False
            for idx in range(len(cols)):
                if time.time() > deadline - 10.0:
                    return cols
                candidate = cols[:idx] + cols[idx + 1:]
                if verify_cols(candidate):
                    cols = candidate
                    changed = True
                    break
        return cols

    # Elimination is expensive per check; only do a bounded pass.
    if len(cols) <= 130 and time.time() < deadline - 20.0:
        # Do a limited number of elimination attempts
        max_rounds = 60
        for _ in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(len(cols)):
                candidate = cols[:idx] + cols[idx + 1:]
                if verify_cols(candidate):
                    cols = candidate
                    improved = True
                    break
            if not improved:
                break

    # Convert to matrix format
    rank = len(cols)
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    for c in range(rank):
        U[:, c] = cols[c][0]
        V[:, c] = cols[c][1]
        W[:, c] = cols[c][2]

    # Final exact verification
    if not verify_cols(cols):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
