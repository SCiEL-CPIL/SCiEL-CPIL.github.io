import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))

    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1

    return U, V, W

def _search_best_placement(n, m, p, max_blocks=4, time_budget=15.0):
    """
    Systematic search over Strassen block placements.
    For each candidate set of non-overlapping 2x2x2 blocks, compute the rank.
    Return the best (U, V, W, rank) found.
    """
    start = time.time()
    best_rank = n * m * p
    best_result = None

    # Precompute all valid single-block placements
    valid_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                valid_blocks.append((i0, j0, k0))

    def blocks_overlap(b1, b2):
        """Check if two 2x2x2 blocks overlap in any (i,j,k) coordinate."""
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if (b1[0]+di, b1[1]+dj, b1[2]+dk) in \
                       {(b2[0]+di2, b2[1]+dj2, b2[2]+dk2) for di2 in range(2) for dj2 in range(2) for dk2 in range(2)}:
                        return True
        return False

    # Try 1 block
    for b in valid_blocks:
        if time.time() - start > time_budget:
            break
        U, V, W = _build_strassen_embedding(n, m, p, [b])
        r = len(U[0])
        if r < best_rank:
            best_rank = r
            best_result = (U, V, W, r)

    # Try 2 blocks
    for idx1 in range(len(valid_blocks)):
        if time.time() - start > time_budget:
            break
        for idx2 in range(idx1 + 1, len(valid_blocks)):
            b1, b2 = valid_blocks[idx1], valid_blocks[idx2]
            if blocks_overlap(b1, b2):
                continue
            U, V, W = _build_strassen_embedding(n, m, p, [b1, b2])
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W, r)

    # Try 3 blocks
    for idx1 in range(len(valid_blocks)):
        if time.time() - start > time_budget:
            break
        for idx2 in range(idx1 + 1, len(valid_blocks)):
            b1 = valid_blocks[idx1]
            b2 = valid_blocks[idx2]
            if blocks_overlap(b1, b2):
                continue
            for idx3 in range(idx2 + 1, len(valid_blocks)):
                b3 = valid_blocks[idx3]
                if blocks_overlap(b1, b3) or blocks_overlap(b2, b3):
                    continue
                U, V, W = _build_strassen_embedding(n, m, p, [b1, b2, b3])
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W, r)

    # Try 4 blocks
    for idx1 in range(len(valid_blocks)):
        if time.time() - start > time_budget:
            break
        for idx2 in range(idx1 + 1, len(valid_blocks)):
            b1 = valid_blocks[idx1]
            b2 = valid_blocks[idx2]
            if blocks_overlap(b1, b2):
                continue
            for idx3 in range(idx2 + 1, len(valid_blocks)):
                b3 = valid_blocks[idx3]
                if blocks_overlap(b1, b3) or blocks_overlap(b2, b3):
                    continue
                for idx4 in range(idx3 + 1, len(valid_blocks)):
                    b4 = valid_blocks[idx4]
                    if blocks_overlap(b1, b4) or blocks_overlap(b2, b4) or blocks_overlap(b3, b4):
                        continue
                    U, V, W = _build_strassen_embedding(n, m, p, [b1, b2, b3, b4])
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = (U, V, W, r)

    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
        best_result = (U, V, W, len(U[0]))

    return best_result

def _try_find_lower_rank(U, V, W, n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to find a decomposition of rank target_rank using least_squares.
    Uses multiple initialization strategies and a final high-precision polish.
    """
    start_time = time.time()

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    n_m = n * m
    m_p = m * p
    p_n = p * n

    def residuals(params):
        U_flat = params[:n_m * target_rank]
        V_flat = params[n_m * target_rank : n_m * target_rank + m_p * target_rank]
        W_flat = params[n_m * target_rank + m_p * target_rank : ]

        U_opt = U_flat.reshape(n_m, target_rank)
        V_opt = V_flat.reshape(m_p, target_rank)
        W_opt = W_flat.reshape(p_n, target_rank)

        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_opt[i*m+j], np.multiply(V_opt[j*p+k], W_opt[k*n+i]))
                    res[i,j,k] = val - 1.0
        return res.flatten()

    # Build list of initializations
    inits = []

    # Init 1: direct truncation
    U0 = U_np[:, :target_rank]
    V0 = V_np[:, :target_rank]
    W0 = W_np[:, :target_rank]
    inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))

    # Init 2: random perturbation
    x0 = inits[0]
    inits.append(x0 + np.random.normal(0, 0.1, len(x0)))

    # Init 3-5: drop different random columns
    orig_rank = U_np.shape[1]
    for drop_idx in range(min(3, orig_rank)):
        mask = [i for i in range(orig_rank) if i != drop_idx]
        if len(mask) >= target_rank:
            U_sub = U_np[:, mask[:target_rank]]
            V_sub = V_np[:, mask[:target_rank]]
            W_sub = W_np[:, mask[:target_rank]]
            inits.append(np.concatenate([U_sub.flatten(), V_sub.flatten(), W_sub.flatten()]))

    # Init 6: random initialization
    np.random.seed(42)
    U_rand = np.random.randn(n_m, target_rank) * 0.5
    V_rand = np.random.randn(m_p, target_rank) * 0.5
    W_rand = np.random.randn(p_n, target_rank) * 0.5
    inits.append(np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()]))

    best_solution = None
    best_cost = float('inf')

    for attempt, x_init in enumerate(inits):
        if time.time() - start_time > time_limit - 2.0:
            break

        try:
            result = least_squares(residuals, x_init, method='lm', max_nfev=800, xtol=1e-9, ftol=1e-9)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
        except Exception:
            pass

    if best_solution is None or best_cost > 1e-6:
        return None

    # High-precision polish pass with tighter tolerance
    if time.time() - start_time < time_limit - 1.0:
        try:
            result2 = least_squares(residuals, best_solution, method='lm', max_nfev=500, xtol=1e-12, ftol=1e-12)
            if result2.cost < best_cost:
                best_cost = result2.cost
                best_solution = result2.x
        except Exception:
            pass

    if best_cost > 1e-8:
        return None

    U_opt = best_solution[:n_m * target_rank].reshape(n_m, target_rank)
    V_opt = best_solution[n_m * target_rank : n_m * target_rank + m_p * target_rank].reshape(m_p, target_rank)
    W_opt = best_solution[n_m * target_rank + m_p * target_rank : ].reshape(p_n, target_rank)

    U_new = U_opt.tolist()
    V_new = V_opt.tolist()
    W_new = W_opt.tolist()

    if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
        return U_new, V_new, W_new

    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS

    # Phase 1: Systematic search over Strassen block placements (from inspiration)
    # This finds the best fixed-placement decomposition
    if _time_left() > 60:
        U, V, W, best_rank = _search_best_placement(n, m, p, max_blocks=4, time_budget=20.0)
    else:
        # Fallback: fixed placement
        blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
        U, V, W = _build_strassen_embedding(n, m, p, blocks)
        best_rank = len(U[0])

    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        best_rank = len(U[0])

    best_result = {"U": U, "V": V, "W": W}

    # Phase 2: Try to reduce rank by 1 via optimization (from current program)
    if _time_left() > 25:
        res = _try_find_lower_rank(U, V, W, n, m, p, best_rank - 1, time_limit=_time_left() - 15)
        if res:
            U_new, V_new, W_new = res
            best_rank = best_rank - 1
            best_result = {"U": U_new, "V": V_new, "W": W_new}

    # Phase 3: Try to reduce rank by another 1 if we found a reduction
    if best_rank > 60 and _time_left() > 15:
        U_cur = best_result["U"]
        V_cur = best_result["V"]
        W_cur = best_result["W"]
        res = _try_find_lower_rank(U_cur, V_cur, W_cur, n, m, p, best_rank - 1, time_limit=_time_left() - 5)
        if res:
            U_new, V_new, W_new = res
            best_rank = best_rank - 1
            best_result = {"U": U_new, "V": V_new, "W": W_new}

    return best_result
