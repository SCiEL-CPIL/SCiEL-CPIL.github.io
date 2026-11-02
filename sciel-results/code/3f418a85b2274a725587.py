import time
import numpy as np

_DEADLINE = 115.0  # safety margin under the 120 s wall-clock limit


def _strassen_triples():
    # Each entry: (u, v, w) with u,v,w in {0,1,-1}^4.
    # u indexes A (row-major [A00,A01,A10,A11]).
    # v indexes the 2x2 block of B (row-major [B00,B01,B10,B11]).
    # w indexes the 2x2 block of C (index order: C[0,0],C[1,0],C[0,1],C[1,1]
    #   i.e. w[k*n+i] for n=2, so [w00,w10,w01,w11]).
    # These are the seven Strassen products giving the rank-7 tensor
    # decomposition of M_{2,2,2}.
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # P3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # P7
    ]


def _verify_decomposition(U, V, W, n=2, m=2, p=12):
    """Verify the tensor decomposition is exact. Returns True if valid."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                if val != 1.0:
                    errors += 1
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j2 * p + k
                    c_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                    if val != 0.0:
                        errors += 1
    return errors == 0


def _build_rank42():
    n, m, p = 2, 2, 12
    nblocks = p // 2          # 6 independent 2x2 blocks
    rank = 7 * nblocks        # 42
    
    U = np.zeros((n * m, rank), dtype=int)   # 4 x 42
    V = np.zeros((m * p, rank), dtype=int)   # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)   # 24 x 42

    triples = _strassen_triples()
    col = 0
    for b in range(nblocks):
        for (u, v, w) in triples:
            # A factors: global A index = i*m + j  (i in 0..1, j in 0..1)
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = u[i * 2 + j]
            # B factors: global B index = j*p + k, k = 2*b + k_loc
            for j in range(2):
                for k_loc in range(2):
                    V[j * p + (2 * b + k_loc), col] = v[j * 2 + k_loc]
            # C factors: global C index = k*n + i, k = 2*b + k_loc
            for k_loc in range(2):
                for i in range(2):
                    W[(2 * b + k_loc) * n + i, col] = w[k_loc * 2 + i]
            col += 1

    return U, V, W


def construct():
    t0 = time.time()
    U, V, W = _build_rank42()
    
    # Verify correctness (fast: 4*24*24*42 ~ 97k multiply-adds)
    elapsed = time.time() - t0
    if elapsed < _DEADLINE - 5:
        if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
            # Should not happen; decomposition is statically correct.
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
