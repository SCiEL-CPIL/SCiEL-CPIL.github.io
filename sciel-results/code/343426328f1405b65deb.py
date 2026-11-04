import numpy as np
import time
from fractions import Fraction

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_M(U, V, W, n, m, p):
    rank = U.shape[1]
    N = n*p*m
    M = np.zeros((N, rank))
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                idx += 1
    return M

def _round_to_half(x):
    return np.round(x * 2) / 2.0

def _try_reduce_general(U, V, W, n, m, p, deadline, max_rounds=200):
    """General reduction: allow any real scaling factor, not just half-integers."""
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(max_rounds):
        if time.time() > deadline - 20.0:
            break
        if R <= 1:
            break

        M = _build_M(U, V, W, n, m, p)

        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break

        tol_base = max(1e-10, 1e-8 * max(s) if len(s) > 0 else 1e-8)
        null_idx = np.where(s < tol_base)[0]
        if len(null_idx) == 0:
            for tol_mult in [10, 100, 1000]:
                null_idx = np.where(s < tol_base * tol_mult)[0]
                if len(null_idx) > 0:
                    break
        if len(null_idx) == 0:
            break

        reduced = False
        for ni in null_idx:
            if time.time() > deadline - 15.0:
                break
            c = Vt[ni, :]
            r0 = int(np.argmax(np.abs(c)))
            if abs(c[r0]) < 1e-12:
                continue

            keep = [q for q in range(R) if q != r0]
            scales = []
            valid = True
            for r in keep:
                factor = 1.0 + c[r] / c[r0]
                if abs(factor) < 1e-14:
                    valid = False
                    break
                if factor >= 0:
                    s_val = factor ** (1.0/3.0)
                else:
                    s_val = -((-factor) ** (1.0/3.0))
                if not np.isfinite(s_val) or abs(s_val) < 1e-15:
                    valid = False
                    break
                scales.append(s_val)

            if not valid:
                continue

            new_U = U[:, keep].copy()
            new_V = V[:, keep].copy()
            new_W = W[:, keep].copy()
            for i, r in enumerate(keep):
                new_U[:, i] *= scales[i]
                new_V[:, i] *= scales[i]
                new_W[:, i] *= scales[i]

            if _verify(new_U, new_V, new_W, n, m, p, tol=1e-5):
                U, V, W = new_U, new_V, new_W
                R -= 1
                reduced = True
                if R < best_rank:
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

        if not reduced:
            break

    return best_U, best_V, best_W, best_rank

def _try_multi_column_reduce(U, V, W, n, m, p, deadline, max_rounds=50):
    """Try to remove multiple columns at once by finding a subspace of the null space."""
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(max_rounds):
        if time.time() > deadline - 25.0:
            break
        if R <= 2:
            break

        M = _build_M(U, V, W, n, m, p)
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break

        tol_base = max(1e-10, 1e-8 * max(s) if len(s) > 0 else 1e-8)
        null_idx = np.where(s < tol_base * 100)[0]
        if len(null_idx) < 2:
            break

        removed = False
        for i in range(len(null_idx)):
            for j in range(i+1, len(null_idx)):
                if time.time() > deadline - 20.0:
                    break
                ci = Vt[null_idx[i], :]
                cj = Vt[null_idx[j], :]
                weights_i = np.abs(ci)
                weights_j = np.abs(cj)
                combined = weights_i + weights_j
                r0 = int(np.argmax(combined))
                if combined[r0] < 1e-10:
                    continue

                keep1 = [q for q in range(R) if q != r0]
                scales1 = []
                valid1 = True
                for r in keep1:
                    factor = 1.0 + ci[r] / ci[r0]
                    if abs(factor) < 1e-14:
                        valid1 = False
                        break
                    if factor >= 0:
                        s_val = factor ** (1.0/3.0)
                    else:
                        s_val = -((-factor) ** (1.0/3.0))
                    if not np.isfinite(s_val):
                        valid1 = False
                        break
                    scales1.append(s_val)

                if not valid1:
                    continue

                new_U1 = U[:, keep1].copy()
                new_V1 = V[:, keep1].copy()
                new_W1 = W[:, keep1].copy()
                for idx, r in enumerate(keep1):
                    new_U1[:, idx] *= scales1[idx]
                    new_V1[:, idx] *= scales1[idx]
                    new_W1[:, idx] *= scales1[idx]

                if not _verify(new_U1, new_V1, new_W1, n, m, p, tol=1e-5):
                    continue

                U, V, W = new_U1, new_V1, new_W1
                R -= 1
                removed = True
                if R < best_rank:
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

        if not removed:
            break

    return best_U, best_V, best_W, best_rank

def _round_final(U, V, W):
    """Round values to nearest half-integer and verify."""
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    return U_r, V_r, W_r

def _try_reduce_phase2(U, V, W, n, m, p, deadline):
    """Second phase: additional reduction with tighter tolerances."""
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(100):
        if time.time() > deadline - 20.0:
            break
        if R <= 1:
            break

        M = _build_M(U, V, W, n, m, p)
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break

        tol_base = max(1e-12, 1e-10 * max(s) if len(s) > 0 else 1e-10)
        null_idx = np.where(s < tol_base)[0]
        if len(null_idx) == 0:
            break

        reduced = False
        for ni in null_idx:
            if time.time() > deadline - 15.0:
                break
            c = Vt[ni, :]
            r0 = int(np.argmax(np.abs(c)))
            if abs(c[r0]) < 1e-14:
                continue

            keep = [q for q in range(R) if q != r0]
            scales = []
            valid = True
            for r in keep:
                factor = 1.0 + c[r] / c[r0]
                if abs(factor) < 1e-16:
                    valid = False
                    break
                if factor >= 0:
                    s_val = factor ** (1.0/3.0)
                else:
                    s_val = -((-factor) ** (1.0/3.0))
                if not np.isfinite(s_val) or abs(s_val) < 1e-17:
                    valid = False
                    break
                scales.append(s_val)

            if not valid:
                continue

            new_U = U[:, keep].copy()
            new_V = V[:, keep].copy()
            new_W = W[:, keep].copy()
            for i, r in enumerate(keep):
                new_U[:, i] *= scales[i]
                new_V[:, i] *= scales[i]
                new_W[:, i] *= scales[i]

            if _verify(new_U, new_V, new_W, n, m, p, tol=1e-6):
                U, V, W = new_U, new_V, new_W
                R -= 1
                reduced = True
                if R < best_rank:
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

        if not reduced:
            break

    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0

    U, V, W = _build(98)
    if not _verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: General SVD-based reduction (allow real scalings)
    U, V, W, rank1 = _try_reduce_general(U, V, W, n, m, p, deadline)

    # Phase 2: Multi-column reduction
    if time.time() < deadline - 30.0:
        U, V, W, rank2 = _try_multi_column_reduce(U, V, W, n, m, p, deadline)
    else:
        rank2 = rank1

    # Phase 3: Tighter tolerance reduction
    if time.time() < deadline - 20.0:
        U, V, W, rank3 = _try_reduce_phase2(U, V, W, n, m, p, deadline)
    else:
        rank3 = rank2

    # Final: Round to half-integers and verify
    U_r, V_r, W_r = _round_final(U, V, W)
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-4):
        U, V, W = U_r, V_r, W_r

    # Final verification
    if not _verify(U, V, W, n, m, p, tol=1e-5):
        U, V, W = _build(98)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
