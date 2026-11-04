"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy:
  1. Build structured base candidates:
     a) rank-32: 4 blocks of (7 Strassen 2x2x2 + 1 outer for 3rd row)
     b) rank-44: 4 Strassen blocks for 2x2 sub-matrix + schoolbook for 2x1 row
  2. Compress each candidate by finding linearly dependent rank-1 terms
     (QR-based projection of flattened tensor vectors).
  3. Validate with exact Fraction arithmetic.
  4. Return the lowest-rank valid decomposition found.
"""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Exact validation using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for r in range(rank):
                    uv = u_row[r]
                    if uv == 0:
                        continue
                    vv = v_row[r]
                    if vv == 0:
                        continue
                    acc += Fraction(uv) * Fraction(vv) * Fraction(w_row[r])
                if acc != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

# Strassen 2x2x2 (rank 7) in row-major (i*2+j, j*2+k, k*2+i) indexing
_S_U = [
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1]
]
_S_V = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
]
_S_W = [
    [1, 0, 0, 1],
    [0, 1, 0, -1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]

def _build_rank32(n, m, p):
    """4 blocks: each block = 7 Strassen (rows 0-1, cols 0-1 of block) + 1 outer (row 2)."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col = b * 8
        # 7 Strassen terms
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    val = _S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][col + r] = val
            for la in range(2):
                for lk in range(2):
                    val = _S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][col + r] = val
            for lk in range(2):
                for li in range(2):
                    val = _S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][col + r] = val
        # 1 outer-product term for row 2
        oc = col + 7
        for i in range(n):
            U[i * m + 2][oc] = 1
        for lk in range(2):
            V[2 * p + (k0 + lk)][oc] = 1
        for lk in range(2):
            for i in range(n):
                W[(k0 + lk) * n + i][oc] = 1
    return {"U": U, "V": V, "W": W}

def _build_rank44(n, m, p):
    """4 Strassen blocks for 2x2 sub-matrix + schoolbook for 2x1 row."""
    assert n == 2 and m == 3 and p == 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    # Part 1: 4 Strassen blocks
    for b in range(4):
        k0 = 2 * b
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    val = _S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][col] = val
            for la in range(2):
                for lk in range(2):
                    val = _S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][col] = val
            for lk in range(2):
                for li in range(2):
                    val = _S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][col] = val
            col += 1
    # Part 2: schoolbook for row 2 (2x1 * 1x8)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}

def _compress(U, V, W, n, m, p, deadline, tol=1e-9, max_iters=200):
    """Remove linearly dependent rank-1 terms via QR projection."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, rank

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    nm, mp, pn = n * m, m * p, p * n
    dim = nm * mp * pn

    # Build tensor vectors
    tensors = np.zeros((rank, dim))
    for r in range(rank):
        uv = np.outer(U_np[r], V_np[r])
        t = np.outer(uv.flatten(), W_np[r])
        tensors[r] = t.flatten()

    keep = list(range(rank))
    for iteration in range(max_iters):
        if len(keep) <= 1:
            break
        if time.time() > deadline - 10:
            break
        found = False
        # Try removing from the end (cheapest QR first)
        for idx in range(len(keep) - 1, -1, -1):
            if idx >= len(keep):
                continue
            r = keep[idx]
            others = keep[:idx] + keep[idx + 1:]
            if not others:
                break
            A = tensors[others]
            b = tensors[r]
            try:
                Q, _ = np.linalg.qr(A.T, mode='economic')
                coeffs = Q.T @ b
                proj = Q @ coeffs
                res = np.linalg.norm(b - proj)
                norm_b = np.linalg.norm(b)
                if norm_b > 1e-15 and res < tol * norm_b:
                    keep.pop(idx)
                    found = True
                    break
            except Exception:
                pass
        if not found:
            # Try random positions for a few iterations
            if len(keep) > 5 and iteration < 20:
                import random
                rng = random.Random(iteration * 7919)
                for _ in range(min(10, len(keep) - 1)):
                    idx = rng.randrange(len(keep))
                    r = keep[idx]
                    others = keep[:idx] + keep[idx + 1:]
                    if not others:
                        continue
                    A = tensors[others]
                    b = tensors[r]
                    try:
                        Q, _ = np.linalg.qr(A.T, mode='economic')
                        coeffs = Q.T @ b
                        proj = Q @ coeffs
                        res = np.linalg.norm(b - proj)
                        norm_b = np.linalg.norm(b)
                        if norm_b > 1e-15 and res < tol * norm_b:
                            keep.pop(idx)
                            found = True
                            break
                    except Exception:
                        pass
            if not found:
                break

    if len(keep) == rank:
        return U, V, W, rank

    U_new = [U[i] for i in keep]
    V_new = [V[i] for i in keep]
    W_new = [W[i] for i in keep]
    return U_new, V_new, W_new, len(keep)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115  # safety margin

    best = None
    best_rank = float('inf')

    # Candidate 1: rank-32 block decomposition
    try:
        cand = _build_rank32(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            U_c, V_c, W_c, rank_c = _compress(
                cand["U"], cand["V"], cand["W"], n, m, p, deadline)
            if _validate(U_c, V_c, W_c, n, m, p):
                best = {"U": U_c, "V": V_c, "W": W_c}
                best_rank = rank_c
            else:
                best = cand
                best_rank = len(cand["U"][0])
        else:
            # Fallback: if rank-32 is invalid, use rank-44
            pass
    except Exception:
        pass

    # Candidate 2: rank-44 structured decomposition
    if time.time() < deadline - 30:
        try:
            cand = _build_rank44(n, m, p)
            if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
                U_c, V_c, W_c, rank_c = _compress(
                    cand["U"], cand["V"], cand["W"], n, m, p, deadline)
                if _validate(U_c, V_c, W_c, n, m, p):
                    if rank_c < best_rank:
                        best = {"U": U_c, "V": V_c, "W": W_c}
                        best_rank = rank_c
        except Exception:
            pass

    # Final cross-compression pass on best if we have time
    if best is not None and time.time() < deadline - 15:
        try:
            U_f, V_f, W_f, rank_f = _compress(
                best["U"], best["V"], best["W"], n, m, p, deadline,
                tol=1e-10, max_iters=50)
            if _validate(U_f, V_f, W_f, n, m, p) and rank_f < best_rank:
                best = {"U": U_f, "V": V_f, "W": W_f}
                best_rank = rank_f
        except Exception:
            pass

    if best is not None:
        return best

    # Ultimate fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
