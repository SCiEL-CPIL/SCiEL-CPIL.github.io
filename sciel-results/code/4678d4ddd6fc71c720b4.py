import numpy as np
import time
import itertools

DIMENSIONS = [5, 5, 5]
_DEADLINE = time.time() + 130.0  # safety margin under 150s limit


def _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col):
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1

    # M1 = (a00+a11)(b00+b11)
    U[i0*m+j0][col] = 1; U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1; V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1; W[k1*n+i1][col] = 1
    col += 1

    # M2 = (a10+a11)b00
    U[i1*m+j0][col] = 1; U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    W[k0*n+i1][col] = 1; W[k1*n+i1][col] = -1
    col += 1

    # M3 = a00(b01-b11)
    U[i0*m+j0][col] = 1
    V[j0*p+k1][col] = 1; V[j1*p+k1][col] = -1
    W[k1*n+i0][col] = 1; W[k1*n+i1][col] = 1
    col += 1

    # M4 = a11(b10-b00)
    U[i1*m+j1][col] = 1
    V[j1*p+k0][col] = 1; V[j0*p+k0][col] = -1
    W[k0*n+i0][col] = 1; W[k0*n+i1][col] = 1
    col += 1

    # M5 = (a00+a01)b11
    U[i0*m+j0][col] = 1; U[i0*m+j1][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = -1; W[k1*n+i0][col] = 1
    col += 1

    # M6 = (a10-a00)(b00+b01)
    U[i1*m+j0][col] = 1; U[i0*m+j0][col] = -1
    V[j0*p+k0][col] = 1; V[j0*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M7 = (a01-a11)(b10+b11)
    U[i0*m+j1][col] = 1; U[i1*m+j1][col] = -1
    V[j1*p+k0][col] = 1; V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    return col


def _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col):
    i0, i1, i2 = i_start, i_start+1, i_start+2
    j0, j1, j2 = j_start, j_start+1, j_start+2
    k0, k1, k2 = k_start, k_start+1, k_start+2

    # M1 = (a00+a22)(b00+b22)
    U[i0*m+j0][col] = 1; U[i2*m+j2][col] = 1
    V[j0*p+k0][col] = 1; V[j2*p+k2][col] = 1
    W[k0*n+i0][col] = 1; W[k2*n+i2][col] = 1
    col += 1

    # M2 = (a11+a22)(b11+b22)
    U[i1*m+j1][col] = 1; U[i2*m+j2][col] = 1
    V[j1*p+k1][col] = 1; V[j2*p+k2][col] = 1
    W[k1*n+i1][col] = 1; W[k2*n+i2][col] = 1
    col += 1

    # M3 = (a00+a11)(b00+b11)
    U[i0*m+j0][col] = 1; U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1; V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1; W[k1*n+i1][col] = 1
    col += 1

    # M4 = (a22+a00)(b22+b00)
    U[i2*m+j2][col] = 1; U[i0*m+j0][col] = 1
    V[j2*p+k2][col] = 1; V[j0*p+k0][col] = 1
    W[k2*n+i2][col] = 1; W[k0*n+i0][col] = 1
    col += 1

    # M5 = (a22+a11)(b22+b11)
    U[i2*m+j2][col] = 1; U[i1*m+j1][col] = 1
    V[j2*p+k2][col] = 1; V[j1*p+k1][col] = 1
    W[k2*n+i2][col] = 1; W[k1*n+i1][col] = 1
    col += 1

    # M6 = (a01+a21)(b10+b20)
    U[i0*m+j1][col] = 1; U[i2*m+j1][col] = 1
    V[j1*p+k0][col] = 1; V[j2*p+k0][col] = 1
    W[k0*n+i0][col] = 1; W[k0*n+i2][col] = 1
    col += 1

    # M7 = (a10+a12)(b01+b21)
    U[i1*m+j0][col] = 1; U[i1*m+j2][col] = 1
    V[j0*p+k1][col] = 1; V[j2*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    return col


def _block_rank(d):
    """Rank to multiply a d x d matrix by a d x d matrix."""
    if d == 2:
        return 7
    if d == 3:
        return 7
    return d * d * d


def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U reshaped to (n, m, rank); V to (m, p, rank); W to (p, n, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-8)


def _build(n, m, p, n_split, m_split, p_split):
    """Build the factor matrices for a given partition.

    n_split, m_split, p_split are lists of block sizes (non-negative) summing to n, m, p.
    """
    total_rank = 0
    for a in n_split:
        for b in m_split:
            for c in p_split:
                total_rank += _block_rank(a) * _block_rank(b) * _block_rank(c)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    i_start = 0
    for a in n_split:
        j_start = 0
        for b in m_split:
            k_start = 0
            for c in p_split:
                if a == 0 or b == 0 or c == 0:
                    pass  # no block
                elif a == 2 and b == 2 and c == 2:
                    col = _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col)
                elif a == 3 and b == 3 and c == 3:
                    col = _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col)
                else:
                    # direct
                    for u in range(a):
                        for v in range(b):
                            for w in range(c):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
                k_start += c
            j_start += b
        i_start += a

    return {"U": U, "V": V, "W": W}, col


def _ordered_splits(total, max_parts=5):
    """Generate all ordered compositions of total into up to max_parts non-negative parts."""
    if total == 0:
        yield [0]
        return
    # Compositions with positive parts, then allow trailing zeros
    def _pos(parts_left, remaining):
        if parts_left == 1:
            yield [remaining]
            return
        for v in range(1, remaining + 1):
            for rest in _pos(parts_left - 1, remaining - v):
                yield [v] + rest

    for num_parts in range(1, max_parts + 1):
        for comp in _pos(num_parts, total):
            yield comp
            # also trailing-zero variants (e.g. [2,3] vs [2,3,0] vs [2,0,3])
            for zpos in range(1, num_parts):
                variant = comp[:zpos] + [0] + comp[zpos:]
                yield variant


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    best_result = None
    best_rank = n * m * p  # direct fallback rank

    # Enumerate ordered partitions of n, m, p
    n_spl = list(_ordered_splits(n, 4))
    m_spl = list(_ordered_splits(m, 4))
    p_spl = list(_ordered_splits(p, 4))

    # Sort each by estimated contribution to keep good candidates early
    def _est_rank(split):
        # cheap upper bound: sum of squares
        return sum(d * d for d in split)

    n_spl.sort(key=_est_rank)
    m_spl.sort(key=_est_rank)
    p_spl.sort(key=_est_rank)

    for n_split in n_spl:
        for m_split in m_spl:
            for p_split in p_spl:
                if time.time() > _DEADLINE:
                    break
                # compute total rank
                total_rank = 0
                for a in n_split:
                    for b in m_split:
                        for c in p_split:
                            total_rank += _block_rank(a) * _block_rank(b) * _block_rank(c)
                if total_rank >= best_rank:
                    continue
                result, col = _build(n, m, p, n_split, m_split, p_split)
                if col != total_rank:
                    continue
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = total_rank

    if best_result is not None:
        return best_result

    # Fallback: direct
    total_rank_direct = n * m * p
    U2 = [[0] * total_rank_direct for _ in range(n * m)]
    V2 = [[0] * total_rank_direct for _ in range(m * p)]
    W2 = [[0] * total_rank_direct for _ in range(p * n)]
    col2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j][col2] = 1
                V2[j*p+k][col2] = 1
                W2[k*n+i][col2] = 1
                col2 += 1
    return {"U": U2, "V": V2, "W": W2}
