import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1.0
                V0[j*p + k, col] = 1.0
                W0[k*n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        lhs = float(np.dot(W[k*n+i, :], U[i*m+j, :] * V[jp*p+k, :]))
                        rhs = 1.0 if j == jp else 0.0
                        if abs(lhs - rhs) > tol:
                            return False
        return True

    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    from scipy.optimize import least_squares

    cur_U, cur_V, cur_W = U0.copy(), V0.copy(), W0.copy()
    r = rank0
    attempts = 0
    max_attempts = 40

    while r > 40 and attempts < max_attempts:
        if time.time() > deadline - 25:
            break
        attempts += 1
        found = False
        for c in range(r):
            if time.time() > deadline - 15:
                break
            r_new = r - 1
            mask = [i for i in range(r) if i != c]
            U_s, V_s, W_s = cur_U[:, mask], cur_V[:, mask], cur_W[:, mask]
            x0 = np.concatenate([U_s.ravel(), V_s.ravel(), W_s.ravel()])

            def res_fn(params, r=r_new):
                Uf = params[:n*m*r].reshape(n*m, r)
                Vf = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                Wf = params[n*m*r + m*p*r :].reshape(p*n, r)
                out = np.empty(n*p*m*m)
                idx = 0
                for i in range(n):
                    for k in range(p):
                        Wi = Wf[k*n+i]
                        for j in range(m):
                            Ui = Uf[i*m+j]
                            for jp in range(m):
                                Vi = Vf[jp*p+k]
                                out[idx] = float(np.dot(Wi, Ui * Vi)) - (1.0 if j == jp else 0.0)
                                idx += 1
                return out

            try:
                sol = least_squares(res_fn, x0, max_nfev=25, xtol=1e-12)
                if sol.cost < 1e-16:
                    nU = sol.x[:n*m*r_new].reshape(n*m, r_new)
                    nV = sol.x[n*m*r_new : n*m*r_new + m*p*r_new].reshape(m*p, r_new)
                    nW = sol.x[n*m*r_new + m*p*r_new :].reshape(p*n, r_new)
                    cur_U, cur_V, cur_W = nU, nV, nW
                    r = r_new
                    found = True
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = nU.copy(), nV.copy(), nW.copy()
                    break
            except Exception:
                pass
        if not found:
            break

    Uo = np.round(cur_U * 2) / 2
    Vo = np.round(cur_V * 2) / 2
    Wo = np.round(cur_W * 2) / 2
    if verify(Uo, Vo, Wo):
        return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}
    if verify(cur_U, cur_V, cur_W):
        return {"U": cur_U.tolist(), "V": cur_V.tolist(), "W": cur_W.tolist()}
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
