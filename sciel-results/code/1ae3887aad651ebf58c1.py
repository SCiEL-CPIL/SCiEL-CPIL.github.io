"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen222():
    # Correct rank-7 Strassen factors for the 2x2x2 scalar tensor.
    # U indexed by (i,j), V by (j,k), W by (k,i), each in 0..3.
    U = np.array([
        [ 1, 0, 1, 0, 1, 1, 0],  # (0,0)
        [ 0, 0, 0, 0, 1, 0, 1],  # (0,1)
        [ 0, 1, 0, 0, 0, 1, 0],  # (1,0)
        [ 1, 1, 0, 1, 0, 0, 1],  # (1,1)
    ], dtype=np.int64)
    V = np.array([
        [ 1, 0, 0, 0, 0, 1, 0],  # (0,0)
        [ 0, 0, 0, 1, 0, 0, 0],  # (0,1)
        [ 1, 1, 1, 0, 1, 0, 1],  # (1,0)
        [ 1, 1, 0, 0, 1, 0, 0],  # (1,1)
    ], dtype=np.int64)
    W = np.array([
        [ 1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [ 0, 0, 0, 0, 1, 0, 0],  # (0,1)
        [ 0, 1, 1, 0, 0, 0, 0],  # (1,0)
        [ 1, 0, 0, 1, 1, 0, 0],  # (1,1)
    ], dtype=np.int64)
    return U, V, W

def _make_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    return T

def _residual(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

def _exact(U, V, W, T):
    return bool(np.array_equal(_residual(U, V, W), T))

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _local_search(U, V, W, T, deadline, max_iters=60000):
    U = U.copy(); V = V.copy(); W = W.copy()
    T = np.asarray(T, dtype=np.int64)
    R = _residual(U, V, W)
    errs = int(np.count_nonzero(R - T))
    if errs == 0:
        return True, (U, V, W)
    best_errs = errs
    best = (U.copy(), V.copy(), W.copy())
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if errs == 0:
            return True, (U, V, W)
        wrong = np.argwhere(R != T)
        if wrong.size == 0:
            break
        a, b, c = wrong[np.random.randint(wrong.shape[0])]
        target_val = int(T[a, b, c])
        cur = int(R[a, b, c])
        need = target_val - cur
        cands = []
        for r in range(U.shape[1]):
            f = int(V[b, r] * W[c, r])
            if f != 0 and need % f == 0:
                new = int(U[a, r]) + need // f
                if abs(new) <= 3 and new != int(U[a, r]):
                    cands.append(('U', a, r, new))
        for r in range(V.shape[1]):
            f = int(U[a, r] * W[c, r])
            if f != 0 and need % f == 0:
                new = int(V[b, r]) + need // f
                if abs(new) <= 3 and new != int(V[b, r]):
                    cands.append(('V', b, r, new))
        for r in range(W.shape[1]):
            f = int(U[a, r] * V[b, r])
            if f != 0 and need % f == 0:
                new = int(W[c, r]) + need // f
                if abs(new) <= 3 and new != int(W[c, r]):
                    cands.append(('W', c, r, new))
        if not cands:
            continue
        mat, idx, r, new = cands[np.random.randint(len(cands))]
        old = None
        if mat == 'U':
            old = int(U[idx, r]); U[idx, r] = new
        elif mat == 'V':
            old = int(V[idx, r]); V[idx, r] = new
        else:
            old = int(W[idx, r]); W[idx, r] = new
        ne = int(np.count_nonzero(_residual(U, V, W) - T))
        if ne < errs:
            errs = ne
            if errs < best_errs:
                best_errs = errs
                best = (U.copy(), V.copy(), W.copy())
        else:
            if mat == 'U': U[idx, r] = old
            elif mat == 'V': V[idx, r] = old
            else: W[idx, r] = old
    return (errs == 0), best

def _build_primary(n, m, p):
    # Split n=3 into 2+1. Use Strassen on the 2x4x4 block (i in 0..1, all j,k in 0..3),
    # schoolbook for i==2.  Strassen needs 7 factors per 2x2x2 sub-block; j,k each have
    # 2 sub-blocks -> 4 sub-blocks -> 28 factors. Remainder i==2: 4*7=28 factors. Total 56.
    sU, sV, sW = _strassen222()
    rank = 28 + (1 * m * p)  # 28 + 28 = 56
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for j0 in range(0, m, 2):
        for k0 in range(0, p, 2):
            for t in range(7):
                for i in range(2):
                    for jj in range(2):
                        j = j0 + jj
                        U[i*m + j, col] += sU[i*2 + jj, t]
                    for kk in range(2):
                        k = k0 + kk
                        for jj in range(2):
                            j = j0 + jj
                            V[j*p + k, col] += sV[jj*2 + kk, t]
                    for kk in range(2):
                        k = k0 + kk
                        for i in range(2):
                            W[k*n + i, col] += sW[kk*2 + i, t]
                col += 1
    for i in range(2, n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    U = U[:, :col]; V = V[:, :col]; W = W[:, :col]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # safety margin before the 150s kill
    T = _make_target(n, m, p)

    # Guaranteed-valid fallback
    sch = _schoolbook(n, m, p)
    assert _exact(sch["U"], sch["V"], sch["W"], T)

    # Primary construction (Strassen 2x4x4 + schoolbook i==2)
    U, V, W = _build_primary(n, m, p)
    if _exact(U, V, W, T):
        return {"U": U, "V": V, "W": W}

    # Refinement: bounded local search to repair / lower the primary construction
    if time.time() < deadline - 5:
        found, (lU, lV, lW) = _local_search(U, V, W, T, deadline)
        if found:
            return {"U": lU, "V": lV, "W": lW}

    # Final safety: guaranteed-valid schoolbook
    return sch

# EVOLVE-BLOCK-END
