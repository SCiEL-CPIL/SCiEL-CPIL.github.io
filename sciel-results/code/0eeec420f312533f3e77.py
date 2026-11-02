"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False

    # Vectorized check: for all i,j,k, sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1
    # Build index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    u_idx = ii * m + jj
    v_idx = jj * p + kk
    w_idx = kk * n + ii
    # Each term: U[u_idx, r] * V[v_idx, r] * W[w_idx, r], summed over r
    # Do it in a loop over r for memory safety
    total = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        total += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
    if np.max(np.abs(total - 1.0)) > 1e-9:
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1

    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _schoolbook_base(bn, bm, bp):
    """Return the schoolbook base decomposition for block sizes bn x bm x bp."""
    rank = bn * bm * bp
    U = np.zeros((bn*bm, rank), dtype=float)
    V = np.zeros((bm*bp, rank), dtype=float)
    W = np.zeros((bp*bn, rank), dtype=float)
    col = 0
    for i in range(bn):
        for j in range(bm):
            for k in range(bp):
                U[i*bm + j, col] = 1
                V[j*bp + k, col] = 1
                W[k*bn + i, col] = 1
                col += 1
    return U, V, W

def _compose_blocks(n, m, p, bn, bm, bp, bU, bV, bW):
    """
    Compose a base decomposition (bn x bm x bp) into an n x m x p decomposition
    by tiling with Kronecker-product structure.
    Requires n % bn == 0, m % bm == 0, p % bp == 0.
    Base rank = bU.shape[1]. Total rank = (n/bn)*(m/bm)*(p/bp) * base_rank.
    """
    if n % bn != 0 or m % bm != 0 or p % bp != 0:
        return None
    ni = n // bn
    nj = m // bm
    nk = p // bp
    base_rank = bU.shape[1]
    total_rank = ni * nj * nk * base_rank

    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)

    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                col_off = (ii * nj + jj) * nk * base_rank + kk * base_rank
                for r in range(base_rank):
                    # U part: bU[li*bm + lj, r] -> U[(ii*bn+li)*m + (jj*bm+lj), col_off+r]
                    for li in range(bn):
                        for lj in range(bm):
                            val = bU[li*bm + lj, r]
                            if val != 0:
                                i = ii*bn + li
                                j = jj*bm + lj
                                U[i*m + j, col_off + r] = val
                    # V part: bV[lj*bp + lk, r] -> V[(jj*bm+lj)*p + (kk*bp+lk), col_off+r]
                    for lj in range(bm):
                        for lk in range(bp):
                            val = bV[lj*bp + lk, r]
                            if val != 0:
                                j = jj*bm + lj
                                k = kk*bp + lk
                                V[j*p + k, col_off + r] = val
                    # W part: bW[lk*bn + li, r] -> W[(kk*bp+lk)*n + (ii*bn+li), col_off+r]
                    for lk in range(bp):
                        for li in range(bn):
                            val = bW[lk*bn + li, r]
                            if val != 0:
                                k = kk*bp + lk
                                i = ii*bn + li
                                W[k*n + i, col_off + r] = val

    def _to_list(M):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in M.tolist()]

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])

    candidates = []

    # Base decompositions to try
    bases = []
    # Strassen 2x2x2 (rank 7)
    bU, bV, bW = _strassen_2x2()
    bases.append((2, 2, 2, bU, bV, bW))

    # Candidate block tilings: (bn, bm, bp)
    tilings = [(2, 2, 2)]

    for (bn, bm, bp, bU, bV, bW) in bases:
        for (tn, tm, tp) in tilings:
            if time.time() > deadline:
                break
            # Block size must divide dimensions
            if n % tn != 0 or m % tm != 0 or p % tp != 0:
                continue
            # The block must be a multiple of the base
            if tn % bn != 0 or tm % bm != 0 or tp % bp != 0:
                continue
            res = _compose_blocks(n, m, p, tn, tm, tp, bU, bV, bW)
            if res is None:
                continue
            rank = len(res["U"][0])
            if rank < best_rank:
                if _verify(res["U"], res["V"], res["W"], n, m, p):
                    best = res
                    best_rank = rank

    return best
# EVOLVE-BLOCK-END
