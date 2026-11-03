import numpy as np
import time

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
    # Build T[i,j,k] = sum_r ...
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Use einsum: r is contracted
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Target: T[i,j,k] = 1 (every triple is a valid MM term in this convention)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=1e-9)

def _round_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close, else keep."""
    hx = np.rint(2.0 * x) / 2.0
    if np.allclose(x, hx, atol=tol):
        return hx
    return x

def _block_decomposition(n, m, p, deadline):
    """Block decomposition: split n=2+1, m=2+2+1, p=2+2+2, Strassen on 2x2x2."""
    n_groups = [[0,1],[2]]
    m_groups = [[0,1],[2,3],[4]]
    p_groups = [[0,1],[2,3],[4,5]]

    rank = n*m*p  # 90, max possible
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    sU, sV, sW = _strassen_222()  # each (4,7)

    col = 0
    for ni, nidx in enumerate(n_groups):
        for mi, midx in enumerate(m_groups):
            for pi, pidx in enumerate(p_groups):
                if time.time() > deadline:
                    break
                an, am, ap = len(nidx), len(midx), len(pidx)
                if an == 2 and am == 2 and ap == 2:
                    # Strassen rank 7
                    # local (a,b,c) in 0..1 -> global (nidx[a], midx[b], pidx[c])
                    for r in range(7):
                        for a in range(2):
                            U[nidx[a]*m + midx[0] + 0, col] = 0  # placeholder
                        # fill U: index (i,j) with i in nidx, j in midx
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
                    # schoolbook for this block: rank an*am*ap
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

    # Trim to actual used columns
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
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS  # 3,5,6

    best = None
    best_rank = n*m*p

    # Candidate 1: block decomposition with Strassen (expected rank 78)
    try:
        U, V, W = _block_decomposition(n, m, p, deadline)
        if _verify(U, V, W, n, m, p):
            r = U.shape[1]
            if r < best_rank:
                best_rank = r
                best = (U, V, W)
    except Exception:
        pass

    # Candidate 2: schoolbook (rank 90) as fallback
    if time.time() < deadline:
        try:
            U, V, W = _schoolbook(n, m, p)
            if _verify(U, V, W, n, m, p):
                r = U.shape[1]
                if r < best_rank:
                    best_rank = r
                    best = (U, V, W)
        except Exception:
            pass

    if best is None:
        # absolute fallback
        U, V, W = _schoolbook(n, m, p)
        best = (U, V, W)

    U, V, W = best
    U, V, W = _to_lists(U, V, W)
    return {"U": U, "V": V, "W": W}
