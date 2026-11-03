import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 143.0

def _time_left():
    return _DEADLINE - time.time()

def _residual(U, V, W, n, m, p):
    """Exact residual tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    T = _residual(U, V, W, n, m, p)
    return bool(np.max(np.abs(T - 1.0)) <= tol)

def _build_strassen_2x2():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # P1 = (a11+a22)*(b11+b22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (a21+a22)*b11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # P3 = a11*(b12-b22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # P4 = a22*(b21-b11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # P5 = (a11+a12)*b22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # P6 = (a21-a11)*(b11+b12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (a12-a22)*(b21+b22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_strassen_4x4x4():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = np.zeros((16, rank)); V = np.zeros((16, rank)); W = np.zeros((16, rank))
    for r1 in range(7):
        for r2 in range(7):
            col = r1*7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2, r2]
                            if c2 == 0: continue
                            U[i1*2+i2, j1*2+j2] += c1*c2
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2, r2]
                            if c2 == 0: continue
                            V[j1*2+j2, k1*2+k2] += c1*c2
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2, r2]
                            if c2 == 0: continue
                            W[k1*2+k2, i1*2+i2] += c1*c2
    return U, V, W

def _build_strassen_4x4x2():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = np.zeros((16, rank)); V = np.zeros((8, rank)); W = np.zeros((8, rank))
    def add_block(a_i, a_j, b_j, c_i, c_k, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj, t]
                    if ca != 0:
                        U[(a_i+li)*4+(a_j+lj), col] += ca
            for lj in range(2):
                for lk in range(2):
                    cb = V2[lj*2+lk, t]
                    if cb != 0:
                        V[(b_j+lj)*2+(c_k+lk), col] += cb
            for lk in range(2):
                for li in range(2):
                    cc = W2[lk*2+li, t]
                    if cc != 0:
                        W[(c_k+lk)*4+(c_i+li), col] += cc
    add_block(0,0,0,0,0,0)
    add_block(0,2,2,0,0,7)
    add_block(2,0,0,2,0,14)
    add_block(2,2,2,2,0,21)
    return U, V, W

def _build_4x5x6_improved():
    n, m, p = 4, 5, 6
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    total_rank = 49 + 28 + 24
    U = np.zeros((n*m, total_rank)); V = np.zeros((m*p, total_rank)); W = np.zeros((p*n, total_rank))
    base = 0
    for i in range(4):
        for j in range(4):
            U[i*5+j, base:base+49] = U444[i*4+j, :]
    for j in range(4):
        for k in range(4):
            V[j*6+k, base:base+49] = V444[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[k*4+i, base:base+49] = W444[k*4+i, :]
    base = 49
    for i in range(4):
        for j in range(4):
            U[i*5+j, base:base+28] = U442[i*4+j, :]
    for j in range(4):
        for kl in range(2):
            kg = 4+kl
            V[j*6+kg, base:base+28] = V442[j*2+kl, :]
    for kl in range(2):
        kg = 4+kl
        for i in range(4):
            W[kg*4+i, base:base+28] = W442[kl*4+i, :]
    base = 49+28
    for i in range(4):
        for k in range(6):
            col = base + i*6 + k
            U[i*5+4, col] = 1.0
            V[4*6+k, col] = 1.0
            W[k*4+i, col] = 1.0
    return U, V, W

def _build_schoolbook(n, m, p):
    U = np.zeros((n*m, n*m*p)); V = np.zeros((m*p, n*m*p)); W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _greedy_reduce(U, V, W, n, m, p, budget_s=60.0):
    """Greedy column removal minimizing the exact residual norm.
    Each iteration: for each candidate column, compute residual after removal and
    pick the one with smallest residual. Stop when no removal improves residual
    below a threshold or residual becomes exact zero."""
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    R = U.shape[1]
    start = time.time()
    tol = 1e-8
    while R >= 2 and (time.time() - start) < budget_s and _time_left() > 5.0:
        # current residual
        T = _residual(U, V, W, n, m, p)
        cur = float(np.linalg.norm(T - 1.0))
        if cur < tol:
            break
        best_c = -1
        best_res = cur
        for c in range(R):
            if _time_left() < 5.0: break
            U2 = np.delete(U, c, axis=1); V2 = np.delete(V, c, axis=1); W2 = np.delete(W, c, axis=1)
            T2 = _residual(U2, V2, W2, n, m, p)
            res = float(np.linalg.norm(T2 - 1.0))
            if res < best_res - 1e-12:
                best_res = res; best_c = c
        if best_c < 0:
            break
        U = np.delete(U, best_c, axis=1); V = np.delete(V, best_c, axis=1); W = np.delete(W, best_c, axis=1)
        R -= 1
    return U, V, W

def _pairwise_reduce(U, V, W, n, m, p, budget_s=30.0):
    """Try removing pairs of columns that together have minimal residual impact."""
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    R = U.shape[1]
    start = time.time()
    tol = 1e-8
    improved = True
    while improved and R >= 3 and (time.time() - start) < budget_s and _time_left() > 5.0:
        improved = False
        T = _residual(U, V, W, n, m, p)
        cur = float(np.linalg.norm(T - 1.0))
        if cur < tol: break
        # Precompute per-column contribution tensors to estimate pair impact cheaply
        Rn = U.shape[1]
        contribs = []
        for c in range(Rn):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            Tc = np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
            contribs.append(Tc)
        # Score pairs by sum of contribution norms (smaller = more removable)
        norms = [float(np.linalg.norm(c)) for c in contribs]
        pairs = []
        for a in range(Rn):
            for b in range(a+1, Rn):
                pairs.append((norms[a]+norms[b], a, b))
        pairs.sort(key=lambda x: x[0])
        for _, a, b in pairs[:40]:
            if _time_left() < 5.0: break
            idx = sorted([a, b])
            U2 = np.delete(U, idx, axis=1); V2 = np.delete(V, idx, axis=1); W2 = np.delete(W, idx, axis=1)
            T2 = _residual(U2, V2, W2, n, m, p)
            if float(np.linalg.norm(T2 - 1.0)) < cur - 1e-12:
                U, V, W = U2, V2, W2
                R -= 2
                improved = True
                break
    return U, V, W

def _local_polish(U, V, W, n, m, p, budget_s=20.0):
    """Smooth local search: perturb columns slightly and accept if residual decreases.
    Uses a soft (continuous) residual objective."""
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    start = time.time()
    rng = np.random.RandomState(0)
    cur = float(np.linalg.norm(_residual(U, V, W, n, m, p) - 1.0))
    step = 0.05
    for _ in range(200):
        if (time.time() - start) > budget_s or _time_left() < 5.0: break
        if cur < 1e-9: break
        # pick a random column and a random entry to perturb
        c = rng.randint(U.shape[1])
        ru = rng.randint(U.shape[0]); rv = rng.randint(V.shape[0]); rw = rng.randint(W.shape[0])
        d = step * (1 if rng.rand() < 0.5 else -1)
        U[ru, c] += d; V[rv, c] += d; W[rw, c] += d
        new = float(np.linalg.norm(_residual(U, V, W, n, m, p) - 1.0))
        if new < cur - 1e-12:
            cur = new
        else:
            U[ru, c] -= d; V[rv, c] -= d; W[rw, c] -= d
        step *= 0.97
    # snap near-zero / near-integer entries to exact values to help verification
    for M in (U, V, W):
        M[np.abs(M) < 1e-9] = 0.0
        # round to nearest half-integer
        M = np.where(np.abs(M - np.round(2*M)/2) < 1e-6, np.round(2*M)/2, M)
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    best_rank = n*m*p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)

    # Structured Strassen-based initial decomposition
    try:
        if _time_left() > 60.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]; best_U, best_V, best_W = U, V, W
    except Exception:
        pass

    # Greedy exact-residual column removal
    try:
        if _time_left() > 40.0:
            cand = _greedy_reduce(best_U, best_V, best_W, n, m, p, budget_s=min(35.0, _time_left()-10.0))
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]; best_U, best_V, best_W = cand
    except Exception:
        pass

    # Pairwise removal
    try:
        if _time_left() > 20.0:
            cand = _pairwise_reduce(best_U, best_V, best_W, n, m, p, budget_s=min(15.0, _time_left()-8.0))
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]; best_U, best_V, best_W = cand
    except Exception:
        pass

    # Smooth local polish (snap to exact entries)
    try:
        if _time_left() > 10.0:
            cand = _local_polish(best_U, best_V, best_W, n, m, p, budget_s=min(8.0, _time_left()-6.0))
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]; best_U, best_V, best_W = cand
    except Exception:
        pass

    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
