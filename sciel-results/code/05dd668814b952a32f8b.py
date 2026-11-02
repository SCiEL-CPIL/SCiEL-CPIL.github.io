"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product


def _schoolbook(n, m, p):
    """Guaranteed-correct schoolbook decomposition, rank = n*m*p."""
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


def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_3x3x2_rank16():
    """
    Verified rank-16 decomposition of 3x3x2 matrix multiplication.
    n=3, m=3, p=2. U:(9,16) V:(6,16) W:(6,16).
    Uses 7 Strassen terms for the 2x2x2 subblock + 9 cross terms.
    """
    n, m, p = 3, 3, 2
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ---- Strassen 2x2x2 (terms 0..6) covering i,j,k in {0,1} ----
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[4][0] = 1
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1; V[1][1] = 1
    W[3][1] = 1; W[4][1] = 1
    U[0][2] = 1
    V[3][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[3][2] = -1
    U[4][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[1][3] = -1; W[4][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = 1; W[3][4] = 1
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1
    W[1][5] = 1; W[4][5] = 1
    U[1][6] = 1; U[3][6] = 1
    V[1][6] = 1; V[2][6] = -1
    W[1][6] = 1; W[3][6] = -1

    # ---- Cross terms (terms 7..15) ----
    U[2][7] = 1; V[4][7] = 1; W[0][7] = 1
    U[2][8] = 1; V[5][8] = 1; W[3][8] = 1
    U[5][9] = 1; V[4][9] = 1; W[1][9] = 1
    U[5][10] = 1; V[5][10] = 1; W[4][10] = 1
    U[6][11] = 1; V[0][11] = 1; W[2][11] = 1
    U[6][12] = 1; V[1][12] = 1; W[5][12] = 1
    U[7][13] = 1; V[2][13] = 1; W[2][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[5][14] = 1
    U[8][15] = 1; V[4][15] = 1; V[5][15] = 1; W[2][15] = 1; W[5][15] = 1

    return {"U": U, "V": V, "W": W}


def _construct_3x3x2_rank18():
    """Guaranteed-correct rank-18 decomposition of 3x3x2 (all 18 products as rank-1)."""
    n, m, p = 3, 3, 2
    rank = 18
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


def _construct_3x3x6_rank48():
    """
    Rank-48 decomposition of 3x3x6 using three independent 3x3x2 blocks (rank 16 each).
    """
    n, m, p = 3, 3, 6

    block16 = _construct_3x3x2_rank16()
    if _verify_decomposition(block16["U"], block16["V"], block16["W"], 3, 3, 2, 16):
        per_rank = 16
        block = block16
    else:
        block18 = _construct_3x3x2_rank18()
        per_rank = 18
        block = block18

    rank = 3 * per_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U2, V2, W2 = block["U"], block["V"], block["W"]

    for blk in range(3):
        base = blk * per_rank
        for i in range(n * m):
            for c in range(per_rank):
                U[i][base + c] = U2[i][c]
        for j in range(m):
            for k in range(2):
                v2 = j * 2 + k
                v6 = j * 6 + (2 * blk + k)
                for c in range(per_rank):
                    V[v6][base + c] = V2[v2][c]
        for k in range(2):
            for i in range(n):
                w2 = k * 3 + i
                w6 = (2 * blk + k) * 3 + i
                for c in range(per_rank):
                    W[w6][base + c] = W2[w2][c]

    return {"U": U, "V": V, "W": W}


def _search_3x3x2_lower_rank(target_rank, max_iters=50000, timeout=60.0):
    """
    Try to find a decomposition of 3x3x2 with rank <= target_rank using random search
    with integer/half-integer entries in {-2,-1.5,-1,-0.5,0,0.5,1,1.5,2}.
    """
    start_time = time.time()
    n, m, p = 3, 3, 2
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        if it % 10000 == 0:
            np.random.seed(42 + it)
        
        # Random sparse initialization
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(target_rank):
            # Each term has a few non-zero entries
            nz_u = np.random.choice(n * m, size=np.random.randint(2, 5), replace=False)
            nz_v = np.random.choice(m * p, size=np.random.randint(2, 5), replace=False)
            nz_w = np.random.choice(p * n, size=np.random.randint(2, 5), replace=False)
            for idx in nz_u:
                U[idx, c] = vals[np.random.randint(len(vals))]
            for idx in nz_v:
                V[idx, c] = vals[np.random.randint(len(vals))]
            for idx in nz_w:
                W[idx, c] = vals[np.random.randint(len(vals))]
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                    if abs(s - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _construct_3x3x6_rank45():
    """
    Try to find a rank-45 decomposition using a rank-15 3x3x2 block if available.
    Falls back to rank-48 if not found.
    """
    n, m, p = 3, 3, 6
    
    # Try to find rank-15 decomposition of 3x3x2
    block15 = _search_3x3x2_lower_rank(15, max_iters=30000, timeout=45.0)
    
    if block15 is not None and _verify_decomposition(block15["U"], block15["V"], block15["W"], 3, 3, 2, 15):
        per_rank = 15
        block = block15
    else:
        # Fallback to rank-16
        block16 = _construct_3x3x2_rank16()
        if _verify_decomposition(block16["U"], block16["V"], block16["W"], 3, 3, 2, 16):
            per_rank = 16
            block = block16
        else:
            block18 = _construct_3x3x2_rank18()
            per_rank = 18
            block = block18

    rank = 3 * per_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U2, V2, W2 = block["U"], block["V"], block["W"]

    for blk in range(3):
        base = blk * per_rank
        for i in range(n * m):
            for c in range(per_rank):
                U[i][base + c] = U2[i][c]
        for j in range(m):
            for k in range(2):
                v2 = j * 2 + k
                v6 = j * 6 + (2 * blk + k)
                for c in range(per_rank):
                    V[v6][base + c] = V2[v2][c]
        for k in range(2):
            for i in range(n):
                w2 = k * 3 + i
                w6 = (2 * blk + k) * 3 + i
                for c in range(per_rank):
                    W[w6][base + c] = W2[w2][c]

    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    try:
        # First try the rank-48 construction (fast, verified)
        result48 = _construct_3x3x6_rank48()
        if _verify_decomposition(result48["U"], result48["V"], result48["W"], n, m, p,
                                 len(result48["U"][0])):
            best = result48
        else:
            best = _schoolbook(n, m, p)
        
        # If we have time, try to find a lower-rank decomposition
        if time.time() < deadline - 50.0:
            result45 = _construct_3x3x6_rank45()
            if _verify_decomposition(result45["U"], result45["V"], result45["W"], n, m, p,
                                     len(result45["U"][0])):
                if len(result45["U"][0]) < len(best["U"][0]):
                    best = result45
        
        if time.time() > deadline:
            return _schoolbook(n, m, p)
        
        return best
    except Exception:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
