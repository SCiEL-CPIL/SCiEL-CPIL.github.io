"""Matrix multiplication 6x6x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _check_fast(Um, Vm, Wm, n, m, p, atol=1e-7):
    """Vectorized check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for every (i,j,k)."""
    R = Um.shape[1]
    Um3 = Um.reshape(n, m, R)
    Vm3 = Vm.reshape(m, p, R)
    Wm3 = Wm.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', Um3, Vm3, Wm3)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook(n, m, p):
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


def _strassen_2x2_terms():
    # Each term: (A_fac, B_fac, C_fac) over the 4 block positions (0,1,2,3)
    return [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]


def _build_block_strassen(n, m, p, block):
    """Apply Strassen 2x2 to a grid of 'block' sized square blocks.
    n = p = block * 2 * grid. Returns (U, V, W) or None."""
    if n % block != 0 or m % block != 0 or p % block != 0:
        return None
    grid = n // block  # number of blocks along each dim
    if grid % 2 != 0:
        return None
    terms = _strassen_2x2_terms()
    total_rank = 7 * grid * grid * grid
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for bi in range(grid):
        for bk in range(grid):
            for bj in range(grid):
                for s in range(7):
                    a_fac, b_fac, c_fac = terms[s]
                    for r in range(2):
                        for c in range(2):
                            idx = r * 2 + c
                            if a_fac[idx] != 0:
                                U[(bi * 2 + r) * block * m + (bk * 2 + c) * block, col] = a_fac[idx]
                            if b_fac[idx] != 0:
                                V[(bk * 2 + r) * block * p + (bj * 2 + c) * block, col] = b_fac[idx]
                            if c_fac[idx] != 0:
                                W[(bj * 2 + c) * block * n + (bi * 2 + r) * block, col] = c_fac[idx]
                    col += 1
    return U, V, W


def _try_remove(Um, Vm, Wm, to_remove, n, m, p, atol=1e-7):
    R = Um.shape[1]
    keep = [r for r in range(R) if r not in to_remove]
    if not keep:
        return None
    return Um[:, keep], Vm[:, keep], Wm[:, keep]


def _greedy_reduce(U, V, W, n, m, p, deadline, atol=1e-7):
    """Greedy single-column removal with random restarts."""
    curU, curV, curW = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    best_rank = rank
    bestU, bestV, bestW = curU.copy(), curV.copy(), curW.copy()
    for restart in range(12):
        if time.time() > deadline:
            break
        cU, cV, cW = U.copy(), V.copy(), W.copy()
        cr = U.shape[1]
        order = list(range(cr))
        random.shuffle(order)
        # repeatedly try to drop the next candidate that still verifies
        progress = True
        while progress:
            if time.time() > deadline:
                break
            progress = False
            for idx in order:
                if time.time() > deadline:
                    break
                if idx >= cr:
                    continue
                cand = _try_remove(cU, cV, cW, {idx}, n, m, p, atol)
                if cand is not None and _check_fast(cand[0], cand[1], cand[2], n, m, p, atol):
                    cU, cV, cW = cand
                    cr -= 1
                    order = [c for c in order if c != idx and c < cr]
                    random.shuffle(order)
                    progress = True
                    if cr < best_rank:
                        best_rank = cr
                        bestU, bestV, bestW = cU.copy(), cV.copy(), cW.copy()
                    break
    return bestU, bestV, bestW, best_rank


def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS

    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    bestU, bestV, bestW = U_sb, V_sb, W_sb
    best_rank = U_sb.shape[1]

    # Candidate structured decompositions (block Strassen)
    candidates = []
    for block in (2, 3):
        res = _build_block_strassen(n, m, p, block)
        if res is not None:
            candidates.append(res)

    # Pick the best structured candidate by verified rank
    for (cU, cV, cW) in candidates:
        if _check_fast(cU, cV, cW, n, m, p):
            r = cU.shape[1]
            if r < best_rank:
                bestU, bestV, bestW = cU, cV, cW
                best_rank = r

    # Greedy rank reduction from the best starting point
    if time.time() < deadline - 15:
        gU, gV, gW, gr = _greedy_reduce(bestU, bestV, bestW, n, m, p, deadline)
        if gr < best_rank and _check_fast(gU, gV, gW, n, m, p):
            bestU, bestV, bestW, best_rank = gU, gV, gW, gr

    # Final safety verification; fall back to schoolbook if anything is off
    if not _check_fast(bestU, bestV, bestW, n, m, p):
        bestU, bestV, bestW, best_rank = U_sb, V_sb, W_sb, U_sb.shape[1]

    return {"U": bestU.tolist(), "V": bestV.tolist(), "W": bestW.tolist()}
# EVOLVE-BLOCK-END
