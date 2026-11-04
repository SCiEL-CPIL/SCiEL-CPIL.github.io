import numpy as np
import time

def construct():
    """
    Rank-minimal search for 4x4 * 4x8 matrix multiplication.

    Structure: build a verified rank-98 decomposition via the Kronecker product
    of two levels of Strassen (2x2x2 rank-7) applied independently to the two
    4x4 column blocks of B. Then greedily remove rank terms that are linear
    combinations of the others in the tensor sense (a term is removable iff its
    rank-1 tensor U_r ⊗ V_r ⊗ W_r lies in the span of the remaining rank-1
    tensors). This is exact for the bilinear tensor and can only lower the rank.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin vs 150s kill

    n, m, p = 4, 4, 8
    nm, mp, pn = n * m, m * p, p * n

    # ---- Strassen 2x2x2 rank-7 decomposition ----
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)

    # ---- Build 4x4x4 rank-49 via Kronecker product ----
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]

    # ---- Build 4x4x8 rank-98 (two independent 4x4 blocks) ----
    rank = 98
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]

    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]

    # ---- Fast exact verification via tensor flattening ----
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  should be 1.
    def tensor_flatten(Um, Vm, Wm, R):
        T = np.zeros((n, m, p), dtype=float)
        for r in range(R):
            T += (Um[:, r] * Wm[:, r][:, None, :]).reshape(n, n, p) \
                 * Vm[:, r][None, :, :]
        return T

    def verify(Um, Vm, Wm, R):
        T = tensor_flatten(Um, Vm, Wm, R)
        return bool(np.allclose(T, 1.0, atol=1e-7))

    # Fallback (schoolbook rank 128)
    def construct_naive():
        U0 = np.zeros((nm, nm*mp), dtype=float)
        V0 = np.zeros((mp, nm*mp), dtype=float)
        W0 = np.zeros((pn, nm*mp), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, idx] = 1.0
                    V0[j*p+k, idx] = 1.0
                    W0[k*n+i, idx] = 1.0
                    idx += 1
        return U0, V0, W0

    if not verify(U, V, W, rank):
        U, V, W = construct_naive()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Greedy rank reduction ----
    # A term r is removable iff its rank-1 tensor T_r lies in the span of the
    # other terms. T_r flattened is a vector of length n*m*p; each term q gives
    # a vector t_q. r is removable iff t_r is in span{t_q : q != r}.
    def removable(r, Um, Vm, Wm, R):
        if R <= 1:
            return False
        # Build matrix of all term tensors (columns), then check if col r is
        # in the span of the others.
        cols = np.empty((n*m*p, R), dtype=float)
        for q in range(R):
            cols[:, q] = (Um[:, q] * Wm[:, q][:, None, :]).reshape(n, n, p) \
                         * Vm[:, q][None, :, :]
        cols = cols.reshape(-1, R)
        others = [q for q in range(R) if q != r]
        A = cols[:, others]
        b = cols[:, r]
        try:
            sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return False
        pred = A @ sol
        if np.linalg.norm(pred - b) > 1e-6 * max(1.0, np.linalg.norm(b)):
            return False
        return True

    U = U.copy(); V = V.copy(); W = W.copy()
    R = U.shape[1]
    removed = []
    while R > 1:
        if time.time() > deadline - 15.0:
            break
        found = False
        for r in range(R):
            if time.time() > deadline - 15.0:
                break
            if removable(r, U, V, W, R):
                keep = [q for q in range(R) if q != r]
                U = U[:, keep]; V = V[:, keep]; W = W[:, keep]
                R -= 1
                removed.append(r)
                found = True
                break
        if not found:
            break

    # Final exact verification; if reduction broke it (shouldn't), revert.
    if R >= 1 and verify(U, V, W, R):
        U = U[:, :R]; V = V[:, :R]; W = W[:, :R]
    else:
        # rebuild clean rank-98
        U = np.zeros((nm, 98), dtype=float); V = np.zeros((mp, 98), dtype=float)
        W = np.zeros((pn, 98), dtype=float)
        U[:, 0:49] = U4; U[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V[j*8+k, 0:49] = V4[j*4+k, 0:49]
                V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0:49] = W4[k*4+i, 0:49]
                W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
