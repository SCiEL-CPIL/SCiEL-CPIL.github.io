import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    t0 = time.time()
    deadline = t0 + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    ncons = n*m*p

    for rank in [8, 9, 10, 11, 12]:
        if time.time() > deadline - 20:
            break
        sol = _find_decomp(n, m, p, rank, deadline)
        if sol is not None:
            U, V, W = sol
            return {"U": U, "V": V, "W": W}

    # Schoolbook fallback
    r = n*m*p
    U = [[0.0]*r for _ in range(nm)]
    V = [[0.0]*r for _ in range(mp)]
    W = [[0.0]*r for _ in range(pn)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _find_decomp(n, m, p, rank, deadline):
    nm, mp, pn = n*m, m*p, p*n
    ncons = n*m*p
    nv = (nm + mp + pn) * rank

    cons_idx = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def residual_and_jac(params):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)

        res = np.zeros(ncons)
        J = np.zeros((ncons, nv))

        idx = 0
        for i, j, k in cons_idx:
            ui = i*m + j
            vi = j*p + k
            wi = k*n + i
            for c in range(rank):
                uf = ui*rank + c
                vf = nm*rank + vi*rank + c
                wf = nm*rank + mp*rank + wi*rank + c
                uc = U[ui, c]
                vc = V[vi, c]
                wc = W[wi, c]
                J[idx, uf] = vc * wc
                J[idx, vf] = uc * wc
                J[idx, wf] = uc * vc
                res[idx] += uc * vc * wc
            res[idx] -= 1.0
            idx += 1

        return res, J

    rng = np.random.RandomState(42)

    for trial in range(200):
        if time.time() > deadline - 5:
            break

        params = rng.randn(nv) * 0.5

        for it in range(300):
            if time.time() > deadline - 5:
                break

            res, J = residual_and_jac(params)
            if np.max(np.abs(res)) < 1e-10:
                U = params[:nm*rank].reshape(nm, rank)
                V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = params[nm*rank+mp*rank:].reshape(pn, rank)
                ok = True
                for i, j, k in cons_idx:
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-6:
                        ok = False
                        break
                if ok:
                    return U.tolist(), V.tolist(), W.tolist()
                break

            try:
                step, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            except:
                break

            params_new = params + step
            res_new, _ = residual_and_jac(params_new)
            if np.linalg.norm(res_new) > 2.0 * np.linalg.norm(res):
                best_params = params
                best_norm = np.linalg.norm(res)
                for damp in [0.5, 0.25, 0.1, 0.05]:
                    p_try = params + damp * step
                    r_try, _ = residual_and_jac(p_try)
                    n_try = np.linalg.norm(r_try)
                    if n_try < best_norm:
                        best_params = p_try
                        best_norm = n_try
                params = best_params
            else:
                params = params_new

    return None
