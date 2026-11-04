"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nU = n * m          # 12
    nV = m * p          # 32
    nW = p * n          # 24
    nEq = n * m * p     # 96
    schoolbook_rank = n * m * p  # 96

    # Equation index maps: (i,j,k) -> (U_idx, V_idx, W_idx)
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eqs.append((i * m + j, j * p + k, k * n + i))
    ui_all = np.array([e[0] for e in eqs], dtype=int)
    vi_all = np.array([e[1] for e in eqs], dtype=int)
    wi_all = np.array([e[2] for e in eqs], dtype=int)

    def tensor_of(U, V, W):
        r = U.shape[1]
        T = np.zeros(nEq, dtype=float)
        for c in range(r):
            T += U[ui_all, c] * V[vi_all, c] * W[wi_all, c]
        return T

    def verify(U, V, W, tol=1e-8):
        T = tensor_of(U, V, W)
        if np.max(np.abs(T - 1.0)) > tol:
            return False
        for M in (U, V, W):
            if not np.allclose(2.0 * M, np.round(2.0 * M), atol=1e-6):
                return False
        return True

    def schoolbook():
        U = np.zeros((nU, schoolbook_rank))
        V = np.zeros((nV, schoolbook_rank))
        W = np.zeros((nW, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best = (U_sb, V_sb, W_sb, schoolbook_rank)

    def halfint_res(M):
        return M - 0.5 * np.round(2.0 * M)

    def obj_grad(x, r, lam):
        U = x[:nU * r].reshape(nU, r)
        V = x[nU * r:nU * r + nV * r].reshape(nV, r)
        W = x[nU * r + nV * r:].reshape(nW, r)
        F = np.zeros(nEq, dtype=float)
        for c in range(r):
            F += U[ui_all, c] * V[vi_all, c] * W[wi_all, c]
        F -= 1.0
        gU = np.zeros((nU, r)); gV = np.zeros((nV, r)); gW = np.zeros((nW, r))
        for c in range(r):
            np.add.at(gU, (ui_all, c), F * V[vi_all, c] * W[wi_all, c])
            np.add.at(gV, (vi_all, c), F * U[ui_all, c] * W[wi_all, c])
            np.add.at(gW, (wi_all, c), F * U[ui_all, c] * V[vi_all, c])
        rU = halfint_res(U); rV = halfint_res(V); rW = halfint_res(W)
        obj = 0.5 * float(np.sum(F * F)) + 0.5 * lam * float(
            np.sum(rU * rU) + np.sum(rV * rV) + np.sum(rW * rW))
        gU = gU + lam * rU
        gV = gV + lam * rV
        gW = gW + lam * rW
        g = np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
        return obj, g

    def adam_solve(r, lam, steps, seed):
        rng = np.random.default_rng(seed)
        size = nU * r + nV * r + nW * r
        x = rng.standard_normal(size) * 0.5
        x = 0.5 * np.round(2.0 * x)
        m_v = np.zeros(size); v_v = np.zeros(size)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = 0.05
        for t in range(1, steps + 1):
            if time.time() > deadline - 5.0:
                break
            obj, g = obj_grad(x, r, lam)
            m_v = b1 * m_v + (1 - b1) * g
            v_v = b2 * v_v + (1 - b2) * (g * g)
            mhat = m_v / (1 - b1 ** t)
            vhat = v_v / (1 - b2 ** t)
            x = x - lr * mhat / (np.sqrt(vhat) + eps)
            if t % 25 == 0:
                Us = 0.5 * np.round(2.0 * x[:nU * r].reshape(nU, r))
                Vs = 0.5 * np.round(2.0 * x[nU * r:nU * r + nV * r].reshape(nV, r))
                Ws = 0.5 * np.round(2.0 * x[nU * r + nV * r:].reshape(nW, r))
                if verify(Us, Vs, Ws, tol=1e-6):
                    return Us, Vs, Ws
        return None

    # Attempt to beat the schoolbook rank with a valid half-integer decomposition.
    for r in range(schoolbook_rank - 1, max(40, schoolbook_rank - 30), -1):
        if time.time() > deadline - 20.0:
            break
        found = None
        for seed in (0, 1):
            if time.time() > deadline - 10.0:
                break
            for lam in (2.0, 10.0):
                if time.time() > deadline - 10.0:
                    break
                res = adam_solve(r, lam, steps=300, seed=seed)
                if res is not None:
                    found = res
                    break
            if found is not None:
                break
        if found is not None:
            U, V, W = found
            if verify(U, V, W, tol=1e-8):
                best = (U, V, W, r)
                continue
        if time.time() > deadline - 15.0:
            break

    U, V, W, r = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
