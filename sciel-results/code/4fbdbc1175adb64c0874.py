import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 6]

def _strassen_222():
    """Return (U,V,W) each shape (4,7) for the 2x2x2 MM tensor (local indices)."""
    U = [
        [1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0],
    ]
    V = [
        [0,1,1,0,0,0,0],
        [0,-1,0,0,0,0,0],
        [1,0,0,0,0,0,0],
        [0,0,0,1,1,0,0],
    ]
    W = [
        [0,1,0,0,0,0,0],
        [1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1],
    ]
    return np.array(U), np.array(V), np.array(W)

def _verify(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] equals the MM tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=1e-9)

def _round_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close, else keep."""
    hx = np.rint(2.0 * x) / 2.0
    if np.allclose(x, hx, atol=tol):
        return hx
    return x

def _all_partitions(d):
    """Generate all ways to partition d into parts of size 1 or 2, as list of sizes."""
    results = []
    def _rec(rem, current):
        if rem == 0:
            results.append(list(current))
            return
        if rem >= 1:
            current.append(1)
            _rec(rem - 1, current)
            current.pop()
        if rem >= 2:
            current.append(2)
            _rec(rem - 2, current)
            current.pop()
    _rec(d, [])
    return results

def _block_cost(an, am, ap):
    """Cost of a block with dimensions (an, am, ap)."""
    if an == 2 and am == 2 and ap == 2:
        return 7
    return an * am * ap

def _total_rank(n_parts, m_parts, p_parts):
    """Compute total rank for given partitions."""
    total = 0
    for an in n_parts:
        for am in m_parts:
            for ap in p_parts:
                total += _block_cost(an, am, ap)
    return total

def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts, deadline):
    """Build factor matrices for given partitions."""
    sU, sV, sW = _strassen_222()
    
    # Precompute index lists for each partition
    def _make_indices(d, parts):
        indices = []
        pos = 0
        for part in parts:
            indices.append(list(range(pos, pos + part)))
            pos += part
        return indices
    
    n_idx = _make_indices(n, n_parts)
    m_idx = _make_indices(m, m_parts)
    p_idx = _make_indices(p, p_parts)
    
    total_rank = _total_rank(n_parts, m_parts, p_parts)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni, nidx in enumerate(n_idx):
        for mi, midx in enumerate(m_idx):
            for pi, pidx in enumerate(p_idx):
                if time.time() > deadline:
                    break
                an, am, ap = len(nidx), len(midx), len(pidx)
                if an == 2 and am == 2 and ap == 2:
                    for r in range(7):
                        for a in range(2):
                            for b in range(2):
                                U[nidx[a]*m + midx[b], col] = sU[a*2+b, r]
                        for b in range(2):
                            for c in range(2):
                                V[midx[b]*p + pidx[c], col] = sV[b*2+c, r]
                        for c in range(2):
                            for a in range(2):
                                W[pidx[c]*n + nidx[a], col] = sW[c*2+a, r]
                        col += 1
                else:
                    for a in range(an):
                        for b in range(am):
                            for c in range(ap):
                                U[nidx[a]*m + midx[b], col] = 1
                                V[midx[b]*p + pidx[c], col] = 1
                                W[pidx[c]*n + nidx[a], col] = 1
                                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    return U, V, W

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    U = _round_half_int(np.asarray(U, dtype=float))
    V = _round_half_int(np.asarray(V, dtype=float))
    W = _round_half_int(np.asarray(W, dtype=float))
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3,5,6

    best = None
    best_rank = n * m * p  # 90

    # Enumerate all valid partitions and find the one with minimum rank
    n_parts_list = _all_partitions(n)
    m_parts_list = _all_partitions(m)
    p_parts_list = _all_partitions(p)

    # Find the best partition combination
    best_partition = None
    best_partition_rank = best_rank

    for np_ in n_parts_list:
        for mp_ in m_parts_list:
            for pp_ in p_parts_list:
                r = _total_rank(np_, mp_, pp_)
                if r < best_partition_rank:
                    best_partition_rank = r
                    best_partition = (np_, mp_, pp_)

    # Build with the best partition
    if best_partition is not None:
        try:
            U, V, W = _build_block_decomposition(n, m, p, *best_partition, deadline)
            if _verify(U, V, W, n, m, p):
                r = U.shape[1]
                if r < best_rank:
                    best_rank = r
                    best = (U, V, W)
        except Exception:
            pass

    # Fallback: schoolbook
    if best is None:
        U, V, W = _schoolbook(n, m, p)
        best = (U, V, W)

    U, V, W = best
    U, V, W = _to_lists(U, V, W)
    return {"U": U, "V": V, "W": W}
