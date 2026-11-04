"""Tensor decomposition for 2x5x5 matrix multiplication.
Uses recursive block splitting with Strassen for 2x2x2 blocks."""
import time
import numpy as np
from itertools import product

N, M, P = 2, 5, 5
_DEADLINE = time.time() + 130  # 20s safety margin


def _schoolbook(n, m, p):
    """Schoolbook decomposition for (n,m,p) matrix mult. Rank = n*m*p."""
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
    return U, V, W


def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2. Rank 7.
    U: 4 rows (a,b,c,d), V: 4 rows (e,f,g,h), W: 4 rows (C00,C01,C10,C11)"""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a+b)(e+f)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1  # C11

    # M3 = a(g+h)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11

    # M4 = d(f-h)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10

    # M5 = (a+d)(e+h)
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[1, 4] = 1  # C01

    # M6 = (b-c)(g+h)
    U[1, 5] = 1; U[2, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1  # C11

    # M7 = (a-c)(e+f)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C00

    return U, V, W


def _decompose_block(n, m, p):
    """Recursively decompose (n,m,p) matrix mult.
    Uses Strassen for 2x2x2, schoolbook for 1-dim blocks,
    and splits larger dimensions as 2+(rest)."""
    if n == 0 or m == 0 or p == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    # Base case: if any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)

    # 2x2x2: use Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()

    # Split the largest dimension into 2 + rest
    # Try different splits and pick the one with minimum rank
    best = None
    best_rank = float('inf')

    def try_split(i_split, j_split, k_split):
        """Split i into i_split+i_rest, etc."""
        i_rest = n - i_split
        j_rest = m - j_split
        k_rest = p - k_split

        # Sub-problems:
        # (i_split, j_split, k_split)
        # (i_split, j_rest, k_split)
        # (i_rest, j_split, k_split)
        # (i_rest, j_rest, k_split)
        # (i_split, j_split, k_rest)
        # (i_split, j_rest, k_rest)
        # (i_rest, j_split, k_rest)
        # (i_rest, j_rest, k_rest)

        sub_problems = [
            (i_split, j_split, k_split),
            (i_split, j_rest, k_split),
            (i_rest, j_split, k_split),
            (i_rest, j_rest, k_split),
            (i_split, j_split, k_rest),
            (i_split, j_rest, k_rest),
            (i_rest, j_split, k_rest),
            (i_rest, j_rest, k_rest),
        ]

        U_list = []
        V_list = []
        W_list = []
        total_rank = 0
        for (ni, nj, nk) in sub_problems:
            if ni > 0 and nj > 0 and nk > 0:
                Ui, Vi, Wi = _decompose_block(ni, nj, nk)
                U_list.append((Ui, ni, nj))
                V_list.append((Vi, nj, nk))
                W_list.append((Wi, nk, ni))
                total_rank += Ui.shape[1]

        return U_list, V_list, W_list, total_rank

    # Try different split strategies
    # Split m (5) into 2+3
    if m >= 3:
        result = try_split(n, 2, p)
        rank = result[3]
        if rank < best_rank:
            best_rank = rank
            best = result

    # Split p (5) into 2+3
    if p >= 3:
        result = try_split(n, m, 2)
        rank = result[3]
        if rank < best_rank:
            best_rank = rank
            best = result

    # Split m into 1+4
    if m >= 2:
        result = try_split(n, 1, p)
        rank = result[3]
        if rank < best_rank:
            best_rank = rank
            best = result

    # Split p into 1+4
    if p >= 2:
        result = try_split(n, m, 1)
        rank = result[3]
        if rank < best_rank:
            best_rank = rank
            best = result

    # Fallback: schoolbook
    if best is None or best_rank >= n * m * p:
        return _schoolbook(n, m, p)

    U_list, V_list, W_list, _ = best

    # Build full matrices
    U_full = np.zeros((n * m, best_rank))
    V_full = np.zeros((m * p, best_rank))
    W_full = np.zeros((p * n, best_rank))

    col = 0
    for idx, (Ui, ni, nj) in enumerate(U_list):
        Vi, nj2, nk = V_list[idx]
        Wi, nk2, ni2 = W_list[idx]
        r = Ui.shape[1]
        # Place Ui into U_full at the correct rows
        # Ui has shape (ni*nj, r), rows correspond to (i,j) in the sub-block
        # But we need to map to the full (n,m) grid
        # The sub-block starts at some offset... 
        # Actually, the recursive decomposition doesn't track offsets.
        # We need a different approach.
        pass

    # The above approach is flawed because we don't track which sub-block
    # each decomposition corresponds to in the full grid.
    # Let's use a cleaner recursive approach with explicit index ranges.
    return _decompose_block_v2(n, m, p, range(n), range(m), range(p))


def _decompose_block_v2(n, m, p, i_range, j_range, k_range):
    """Decompose (n,m,p) with explicit index ranges for placement."""
    if n == 0 or m == 0 or p == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)

    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()

    # Split the middle dimension (m) if it's >= 3
    best_rank = float('inf')
    best_result = None

    # Strategy: split m into 2 + (m-2)
    if m >= 3:
        j_first = j_range[:2]
        j_second = j_range[2:]
        
        # Sub-problems:
        # 1: (n, 2, p)  - i_range, j_first, k_range
        # 2: (n, m-2, p) - i_range, j_second, k_range
        
        U1, V1, W1 = _decompose_block_v2(n, len(j_first), p, i_range, j_first, k_range)
        U2, V2, W2 = _decompose_block_v2(n, len(j_second), p, i_range, j_second, k_range)
        
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = (U1, V1, W1, i_range, j_first, k_range,
                          U2, V2, W2, i_range, j_second, k_range)

    # Strategy: split p into 2 + (p-2)
    if p >= 3:
        k_first = k_range[:2]
        k_second = k_range[2:]
        
        U1, V1, W1 = _decompose_block_v2(n, m, len(k_first), i_range, j_range, k_first)
        U2, V2, W2 = _decompose_block_v2(n, m, len(k_second), i_range, j_range, k_second)
        
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = (U1, V1, W1, i_range, j_range, k_first,
                          U2, V2, W2, i_range, j_range, k_second)

    # Strategy: split m into 1 + (m-1)
    if m >= 2:
        j_first = j_range[:1]
        j_second = j_range[1:]
        
        U1, V1, W1 = _decompose_block_v2(n, 1, p, i_range, j_first, k_range)
        U2, V2, W2 = _decompose_block_v2(n, m-1, p, i_range, j_second, k_range)
        
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = (U1, V1, W1, i_range, j_first, k_range,
                          U2, V2, W2, i_range, j_second, k_range)

    # Strategy: split p into 1 + (p-1)
    if p >= 2:
        k_first = k_range[:1]
        k_second = k_range[1:]
        
        U1, V1, W1 = _decompose_block_v2(n, m, 1, i_range, j_range, k_first)
        U2, V2, W2 = _decompose_block_v2(n, m, p-1, i_range, j_range, k_second)
        
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = (U1, V1, W1, i_range, j_range, k_first,
                          U2, V2, W2, i_range, j_range, k_second)

    # Fallback: schoolbook
    if best_result is None or best_rank >= n * m * p:
        return _schoolbook(n, m, p)

    (U1, V1, W1, i1, j1, k1,
     U2, V2, W2, i2, j2, k2) = best_result

    # Combine into full-sized matrices
    U_full = np.zeros((n * m, best_rank))
    V_full = np.zeros((m * p, best_rank))
    W_full = np.zeros((p * n, best_rank))

    col = 0
    # Place first sub-problem
    for idx_u, (ii, jj) in enumerate(product(i1, j1)):
        U_full[ii * m + jj, col:col+U1.shape[1]] = U1[idx_u, :]
    for idx_v, (jj, kk) in enumerate(product(j1, k1)):
        V_full[jj * p + kk, col:col+V1.shape[1]] = V1[idx_v, :]
    for idx_w, (kk, ii) in enumerate(product(k1, i1)):
        W_full[kk * n + ii, col:col+W1.shape[1]] = W1[idx_w, :]
    col += U1.shape[1]

    # Place second sub-problem
    for idx_u, (ii, jj) in enumerate(product(i2, j2)):
        U_full[ii * m + jj, col:col+U2.shape[1]] = U2[idx_u, :]
    for idx_v, (jj, kk) in enumerate(product(j2, k2)):
        V_full[jj * p + kk, col:col+V2.shape[1]] = V2[idx_v, :]
    for idx_w, (kk, ii) in enumerate(product(k2, i2)):
        W_full[kk * n + ii, col:col+W2.shape[1]] = W2[idx_w, :]

    return U_full, V_full, W_full


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition correctness."""
    rank = U.shape[1]
    S = np.zeros((n, m, p))
    for s in range(rank):
        U_s = U[:, s].reshape(n, m)
        V_s = V[:, s].reshape(m, p)
        W_s = W[:, s].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i, j, k] += U_s[i, j] * V_s[j, k] * W_s[k, i]
    return np.allclose(S, 1.0, atol=tol)


def construct():
    start_time = time.time()
    n, m, p = N, M, P

    # Get recursive decomposition
    U, V, W = _decompose_block_v2(n, m, p, range(n), range(m), range(p))
    rank = U.shape[1]

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = n * m * p

    elapsed = time.time() - start_time
    if elapsed > 110:
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
