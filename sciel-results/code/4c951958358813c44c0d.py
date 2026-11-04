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

def _is_half_integer(x, tol=1e-4):
    y = x * 2
    return abs(y - round(y)) < tol

def _try_reduce(U, V, W, n, m, p, deadline, max_rounds=50):
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(max_rounds):
        if time.time() > deadline - 10.0:
            break
        if R <= 1:
            break

        M = _build_M(U, V, W, n, m, p)
        reduced = False

        # Strategy 1: Remove columns whose tensor product is exactly zero
        for r0 in range(R):
            if time.time() > deadline - 5.0:
                break
            if np.linalg.norm(M[:, r0]) < 1e-10:
                keep = [q for q in range(R) if q != r0]
                new_U = U[:, keep].copy()
                new_V = V[:, keep].copy()
                new_W = W[:, keep].copy()
                if _verify(new_U, new_V, new_W, n, m, p):
                    U, V, W = new_U, new_V, new_W
                    R -= 1
                    reduced = True
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break

        if reduced:
            continue

        # Strategy 2: Merge two columns whose tensor products are scalar multiples
        for r0 in range(R):
            if time.time() > deadline - 5.0:
                break
            for r1 in range(r0+1, R):
                if time.time() > deadline - 5.0:
                    break
                n0 = np.linalg.norm(M[:, r0])
                n1 = np.linalg.norm(M[:, r1])
                if n0 < 1e-10 or n1 < 1e-10:
                    continue
                # Check if M[:,r0] = c * M[:,r1]
                c = np.dot(M[:, r0], M[:, r1]) / (n1 * n1)
                residual = M[:, r0] - c * M[:, r1]
                if np.linalg.norm(residual) < 1e-8 * max(n0, 1.0):
                    # They are proportional with ratio c
                    # We can merge: keep only r1, and we need the combined tensor product
                    # T_new = T_{r0} + T_{r1} = (1+c) * T_{r1}
                    # Scale column r1 by s where s^3 = 1+c
                    # s must be half-integer
                    s_cubed = 1.0 + c
                    if s_cubed < 0:
                        s_val = -np.cbrt(abs(s_cubed))
                    else:
                        s_val = np.cbrt(s_cubed)
                    if _is_half_integer(s_val):
                        keep = [q for q in range(R) if q != r0]
                        # Find index of r1 in keep
                        idx_r1 = keep.index(r1)
                        new_U = U[:, keep].copy()
                        new_V = V[:, keep].copy()
                        new_W = W[:, keep].copy()
                        new_U[:, idx_r1] *= s_val
                        new_V[:, idx_r1] *= s_val
                        new_W[:, idx_r1] *= s_val
                        if _verify(new_U, new_V, new_W, n, m, p):
                            U, V, W = new_U, new_V, new_W
                            R -= 1
                            reduced = True
                            if R < best_rank:
                                best_rank = R
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            break
            if reduced:
                break

        # Strategy 3: Try removing a column that is in the span of others,
        # checking if the remaining set still decomposes the tensor.
        # This works when the removed column's contribution is redundant.
        for r0 in range(R):
            if time.time() > deadline - 5.0:
                break
            keep = [q for q in range(R) if q != r0]
            M_keep = M[:, keep]
            try:
                Q, R_mat = np.linalg.qr(M_keep)
                proj = Q @ (Q.T @ M[:, r0])
                residual = M[:, r0] - proj
                if np.linalg.norm(residual) < 1e-8:
                    # Try removing without scaling
                    new_U = U[:, keep].copy()
                    new_V = V[:, keep].copy()
                    new_W = W[:, keep].copy()
                    if _verify(new_U, new_V, new_W, n, m, p):
                        U, V, W = new_U, new_V, new_W
                        R -= 1
                        reduced = True
                        if R < best_rank:
                            best_rank = R
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
            except np.linalg.LinAlgError:
                pass

        if not reduced:
            break

    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0

    U, V, W = _build(98)
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"

    U, V, W, rank = _try_reduce(U, V, W, n, m, p, deadline)

    if not _verify(U, V, W, n, m, p, tol=1e-5):
        U, V, W = _build(98)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
