"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    
    # Check all on-diagonal entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all off-diagonal entries equal 0
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    val = 0.0
                    for cc in range(rank):
                        val += U[a][cc] * V[b][cc] * W[c][cc]
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _als_solve(U, V, W, n=2, m=5, p=6, max_iter=80, tol=1e-9, deadline=None):
    """ALS for tensor decomposition. Returns (U, V, W, success)."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    U, V, W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        
        # Solve for W
        KR = np.zeros((nm * mp, rank))
        for a in range(nm):
            for b in range(mp):
                KR[a * mp + b] = U[a] * V[b]
        W_new = np.zeros((pn, rank))
        for c in range(pn):
            tgt = np.zeros(nm * mp)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if k * n + i == c:
                            tgt[(i * m + j) * mp + (j * p + k)] = 1.0
            try:
                W_new[c] = np.linalg.lstsq(KR.T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        W = W_new
        
        # Solve for V
        KR = np.zeros((pn * nm, rank))
        for c in range(pn):
            for a in range(nm):
                KR[c * nm + a] = W[c] * U[a]
        V_new = np.zeros((mp, rank))
        for b in range(mp):
            tgt = np.zeros(pn * nm)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if j * p + k == b:
                            tgt[(k * n + i) * nm + (i * m + j)] = 1.0
            try:
                V_new[b] = np.linalg.lstsq(KR.T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        V = V_new
        
        # Solve for U
        KR = np.zeros((mp * pn, rank))
        for b in range(mp):
            for c in range(pn):
                KR[b * pn + c] = V[b] * W[c]
        U_new = np.zeros((nm, rank))
        for a in range(nm):
            tgt = np.zeros(mp * pn)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m + j == a:
                            tgt[(j * p + k) * pn + (k * n + i)] = 1.0
            try:
                U_new[a] = np.linalg.lstsq(KR.T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        U = U_new
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i * m + j, j * p + k, k * n + i
                    err = max(err, abs(np.dot(U[a] * V[b], W[c]) - 1.0))
        if err < tol:
            off = 0.0
            for a in range(nm):
                ii, jj = a // m, a % m
                for b in range(mp):
                    jj2, kk = b // p, b % p
                    if jj != jj2: continue
                    for c in range(pn):
                        kk2, ii2 = c // n, c % n
                        if kk != kk2 or ii != ii2:
                            off = max(off, abs(np.dot(U[a] * V[b], W[c])))
            if off < tol:
                return U, V, W, True
    return U, V, W, False

def _try_rank(r, n=2, m=5, p=6, n_trials=2, deadline=None):
    """Try rank-r decomposition via ALS."""
    nm, mp, pn = n * m, m * p, p * n
    for _ in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        U, V, W, ok = _als_solve(U, V, W, n, m, p, 60, 1e-8, deadline)
        if ok:
            res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify_rank(res["U"], res["V"], res["W"], n, m, p):
                return res
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 130
    n, m, p = DIMENSIONS
    fb = _schoolbook_rank()
    for r in [59, 58]:
        if time.time() > deadline:
            break
        res = _try_rank(r, n, m, p, 2, deadline)
        if res is not None:
            return res
    return fb

# EVOLVE-BLOCK-END
