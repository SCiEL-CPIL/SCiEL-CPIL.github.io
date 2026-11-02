import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 110.0  # safety margin before the 150 s kill


def _build_tensor(n, m, p):
    # T[out, u, v] where out = k*n+i (W convention), u = i*m+j, v = j*p+k
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    return T


def _flattening_lower_bound(T):
    I, J, K = T.shape
    r1 = np.linalg.matrix_rank(T.reshape(I, J * K))
    r2 = np.linalg.matrix_rank(T.transpose(0, 2, 1).reshape(J, I * K))
    r3 = np.linalg.matrix_rank(T.transpose(1, 0, 2).reshape(K, I * J))
    return int(max(r1, r2, r3, max(T.shape)))


def _solve_gram(G, b):
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, b, rcond=None)[0]


def _als_decompose(T, r, n_iter=80, seed=0, tol=1e-11):
    I, J, K = T.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((I, r))
    V = rng.standard_normal((J, r))
    W = rng.standard_normal((K, r))
    for t in range(r):
        nrm = max(1e-12, np.sqrt(np.sum(U[:, t] ** 2) + np.sum(V[:, t] ** 2) + np.sum(W[:, t] ** 2)))
        U[:, t] /= nrm
        V[:, t] /= nrm
        W[:, t] /= nrm
    best_err = np.inf
    for it in range(n_iter):
        # update U rows: (V⊙W) U[i,:]^T = T[i,:,:]
        VW = np.einsum('jt,kt->jkt', V, W).reshape(J * K, r)
        G = VW.T @ VW
        for i in range(I):
            rhs = T[i, :, :].reshape(-1)
            b = VW.T @ rhs
            U[i, :] = _solve_gram(G, b)
        WV = np.einsum('kt,jt->jkt', W, V).reshape(J * K, r)
        G = WV.T @ WV
        for j in range(J):
            rhs = T[:, j, :].reshape(-1)
            b = WV.T @ rhs
            V[j, :] = _solve_gram(G, b)
        UV = np.einsum('it,jt->ijt', U, V).reshape(I * J, r)
        G = UV.T @ UV
        for k in range(K):
            rhs = T[:, :, k].reshape(-1)
            b = UV.T @ rhs
            W[k, :] = _solve_gram(G, b)
        recon = np.einsum('it,jt,kt->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
        if err < tol:
            break
    if best_err < 1e-7:
        return U, V, W, best_err
    return None


def _snap_and_verify(U, V, W, T, atol=1e-6):
    for scale in (2, 1):  # half-integers then integers
        Us = np.round(U * scale) / scale
        Vs = np.round(V * scale) / scale
        Ws = np.round(W * scale) / scale
        recon = np.einsum('it,jt,kt->ijk', Us, Vs, Ws)
        if np.allclose(recon, T, atol=atol):
            return Us, Vs, Ws
    return None


def _to_list(M):
    out = []
    for row in M:
        newrow = []
        for x in row:
            v = round(2 * x) / 2.0
            if abs(v - round(v)) < 1e-6:
                v = float(round(v))
            newrow.append(v)
        out.append(newrow)
    return out


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    lb = _flattening_lower_bound(T)
    best = None
    max_rank = n * m * p
    for r in range(lb, max_rank + 1):
        if time.time() - start > DEADLINE:
            break
        found = None
        for seed in range(4):
            if time.time() - start > DEADLINE:
                break
            res = _als_decompose(T, r, n_iter=80, seed=seed)
            if res is not None:
                U, V, W, err = res
                snapped = _snap_and_verify(U, V, W, T)
                if snapped is not None:
                    found = snapped
                    break
        if found is not None:
            best = (r, found)
            break
    if best is None:
        r = max_rank
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best = (r, (U, V, W))
    r, (U, V, W) = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
