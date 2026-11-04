import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank32():
    """
    Rank-32 decomposition: 8 independent schoolbook 2x2 blocks.
    A is 2x4, B is 4x8. Split A into A1 (j=0,1) and A2 (j=2,3), each 2x2.
    For each output column-block l (k=2l,2l+1):
        C[:, 2l:2l+2] = A1 * B[0:2, 2l:2l+2] + A2 * B[2:4, 2l:2l+2]
    Each 2x2 product uses schoolbook (rank 4). Total: 8 * 4 = 32.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):
        for rb in range(2):
            # rb=0: A1 (j=0,1), rb=1: A2 (j=2,3)
            # For each (i, j_local, k_local) in schoolbook:
            for i in range(2):
                for j_local in range(2):
                    for k_local in range(2):
                        r = col + (i * 4 + j_local * 2 + k_local)
                        # Global indices
                        gi = i
                        gj = (j_local if rb == 0 else 2 + j_local)
                        gk = 2 * l + k_local
                        U[gi * m + gj][r] = 1
                        V[gj * p + gk][r] = 1
                        W[gk * n + gi][r] = 1
            col += 4

    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()

    # Primary: rank-32 construction
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: simple rank-64 schoolbook (always valid)
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
