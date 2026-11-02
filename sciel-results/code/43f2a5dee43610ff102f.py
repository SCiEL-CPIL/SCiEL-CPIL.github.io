"""Matrix multiplication tensor decomposition for 4x4 * 4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify_decomp(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_4x4x4():
    """
    Construct a rank-56 decomposition for 4x4*4x4 using 8 Strassen 2x2 blocks.
    Returns (U4, V4, W4) each of shape (16, 56).
    """
    rank = 56
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))

    def add_block(a, b, c, start_col):
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
        # M2 = (A21+A22)B11
        terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]))
        # M3 = A11(B21-B22)
        terms.append(([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]))
        # M4 = A22(B12-B11)
        terms.append(([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]))
        # M5 = (A11+A12)B22
        terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]))
        # M6 = (A21-A11)(B11+B12)
        terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))
        # M7 = (A12-A22)(B21+B22)
        terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))

        col = start_col
        for (u_entries, v_entries, w_entries) in terms:
            for (ra, ca, coeff) in u_entries:
                U4[(2*a+ra)*4 + (2*b+ca), col] += coeff
            for (rb, cb, coeff) in v_entries:
                V4[(2*b+rb)*4 + (2*c+cb), col] += coeff
            for (rc, ca, coeff) in w_entries:
                W4[(2*c+rc)*4 + (2*a+ca), col] += coeff
            col += 1
        return col

    col = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                col = add_block(a, b, c, col)
    return U4, V4, W4

def _build_4x4x8_from_4x4x4(U4, V4, W4, r4):
    """
    Given a rank-r4 decomposition for 4x4*4x4, build a rank-2*r4 decomposition
    for 4x4*4x8 by applying it to two 4x4 halves of the 8-column B.
    """
    n, m, p = 4, 4, 8
    rank = 2 * r4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U[:, :r4] = U4
    U[:, r4:] = U4

    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :r4] = V4[j * 4 + k]
            V[j * 8 + (k + 4), r4:] = V4[j * 4 + k]

    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :r4] = W4[k * 4 + i]
            W[(k + 4) * 4 + i, r4:] = W4[k * 4 + i]

    return U, V, W

def _try_rank_dense(n, m, p, rank, rng, max_iters, time_limit=None):
    """
    Try to find a rank-`rank` exact decomposition using dense random U,V.
    For each (i,k) pair, solve a linear system for W[k*n+i, :].
    """
    num_u = n * m
    num_v = m * p
    num_w = p * n
    start = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Dense random U, V
        U = rng.standard_normal((num_u, rank))
        V = rng.standard_normal((num_v, rank))
        
        W = np.zeros((num_w, rank))
        ok = True
        for i in range(n):
            for k in range(p):
                w_row = k * n + i
                A_rows = []
                for j in range(m):
                    A_rows.append(U[i * m + j] * V[j * p + k])
                A_mat = np.array(A_rows, dtype=float)  # (m, rank)
                b_vec = np.ones(m, dtype=float)
                try:
                    sol, residuals, rnk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                W[w_row, :] = sol
            if not ok:
                break
        if not ok:
            continue
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(s - 1.0) > 1e-7:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U, V, W
    return None

def _try_rank_4x4x4(rank, rng, max_iters, time_limit=None):
    """Try to find a rank-`rank` decomposition for 4x4*4x4."""
    n = m = p = 4
    num_u = n * m
    num_v = m * p
    num_w = p * n
    start = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        U4 = rng.standard_normal((num_u, rank))
        V4 = rng.standard_normal((num_v, rank))
        
        W4 = np.zeros((num_w, rank))
        ok = True
        for i in range(n):
            for k in range(p):
                w_row = k * n + i
                A_rows = []
                for j in range(m):
                    A_rows.append(U4[i * m + j] * V4[j * m + k])
                A_mat = np.array(A_rows, dtype=float)
                b_vec = np.ones(m, dtype=float)
                try:
                    sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                W4[w_row, :] = sol
            if not ok:
                break
        if not ok:
            continue
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U4[i * m + j] * V4[j * m + k] * W4[k * n + i])
                    if abs(s - 1.0) > 1e-7:
                        valid = False
                        break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U4, V4, W4
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Start with guaranteed rank-112 (2 * rank-56 Strassen blocks)
    U4, V4, W4 = _strassen_4x4x4()
    r4 = 56
    U, V, W = _build_4x4x8_from_4x4x4(U4, V4, W4, r4)
    if _verify_decomp(U, V, W, n, m, p):
        best_rank = 2 * r4
        best = (best_rank, U.copy(), V.copy(), W.copy())
    else:
        sb = _schoolbook(n, m, p)
        return sb

    # Try to find rank-49 for 4x4x4, which gives rank-98 for 4x4x8
    if time.time() < deadline - 30:
        rng49 = np.random.default_rng(12345)
        found49 = _try_rank_4x4x4(49, rng49, max_iters=20000, time_limit=deadline - 30)
        if found49 is not None:
            U4t, V4t, W4t = found49
            U98, V98, W98 = _build_4x4x8_from_4x4x4(U4t, V4t, W4t, 49)
            if _verify_decomp(U98, V98, W98, n, m, p):
                best_rank = 98
                best = (best_rank, U98.copy(), V98.copy(), W98.copy())

    # Try direct rank-r search for 4x4x8 with r < best_rank
    if time.time() < deadline - 15:
        rng_direct = np.random.default_rng(777)
        for target_rank in range(best_rank - 1, 95, -1):
            if time.time() > deadline - 15:
                break
            # Allocate remaining time proportionally
            remaining = deadline - 15 - time.time()
            max_iters = min(10000, max(1000, int(remaining * 50)))
            result = _try_rank_dense(n, m, p, target_rank, rng_direct, max_iters, time_limit=deadline - 15)
            if result is not None:
                U_t, V_t, W_t = result
                if _verify_decomp(U_t, V_t, W_t, n, m, p):
                    best_rank = target_rank
                    best = (best_rank, U_t.copy(), V_t.copy(), W_t.copy())
                    # Continue trying lower ranks
                    rng_direct = np.random.default_rng(target_rank * 1000 + 7)

    _, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
