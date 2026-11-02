import numpy as np
import scipy.optimize as opt
import time

n, m, p = 2, 4, 5
nm, mp, pn = n * m, m * p, p * n


def construct():
    start = time.time()
    deadline = 140.0

    def time_left():
        return deadline - time.time()

    # Target tensor T0[i,k,a,b,c,d] = 1 iff a==i and b==c and d==k
    T0 = np.zeros((n, p, n, m, m, p))
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            if a == i and b == c and d == k:
                                T0[i, k, a, b, c, d] = 1.0

    def residual(x, R):
        sU = nm * R
        sV = mp * R
        U = x[0:sU].reshape(n, m, R)
        V = x[sU:sU + sV].reshape(m, p, R)
        W = x[sU + sV:].reshape(p, n, R)
        T = np.einsum('abm,cdm,kim->ikabcd', U, V, W)
        return (T - T0).ravel()

    def verify(U, V, W):
        R = U.shape[1]
        T = np.einsum('abm,cdm,kim->ikabcd', U, V, W)
        return np.allclose(T, T0, atol=1e-9)

    def snap(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def build_rank36():
        R = 36
        U = np.zeros((nm, R)); V = np.zeros((mp, R)); W = np.zeros((pn, R))
        col = 0

        def strassen(i0, j0, k0):
            nonlocal col
            U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1; col += 1
            U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1
            W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1; col += 1
            U[i0*m+j0, col] += 1
            V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
            W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1; col += 1
            U[(i0+1)*m+(j0+1), col] += 1
            V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
            W[k0*n+i0, col] += 1; W[k0*n+(i0+1), col] += 1; col += 1
            U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1; col += 1
            U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
            V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
            W[(k0+1)*n+(i0+1), col] += 1; col += 1
            U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
            V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1; col += 1

        def school(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m+(j0+dj), col] = 1
                    V[(j0+dj)*p+k0, col] = 1
                    W[k0*n+(i0+di), col] = 1
                    col += 1

        strassen(0, 0, 0); strassen(0, 2, 0); strassen(0, 0, 2); strassen(0, 2, 2)
        school(0, 0, 4); school(0, 2, 4)
        return U, V, W

    U36, V36, W36 = build_rank36()
    if not verify(U36, V36, W36):
        R = nm * mp
        U = np.zeros((nm, R)); V = np.zeros((mp, R)); W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1; col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U36, V36, W36 = snap(U36, V36, W36)
    best = (U36, V36, W36)

    def try_rank(R, ntrials):
        for t in range(ntrials):
            if time_left() < 3:
                break
            if t == 0:
                U0 = U36[:, :R] if R < 36 else U36.copy()
                V0 = V36[:, :R] if R < 36 else V36.copy()
                W0 = W36[:, :R] if R < 36 else W36.copy()
            elif t == 1:
                U0 = U36[:, :R] + 0.15 * np.random.randn(nm, R) if R < 36 else U36 + 0.15 * np.random.randn(nm, R)
                V0 = V36[:, :R] + 0.15 * np.random.randn(mp, R) if R < 36 else V36 + 0.15 * np.random.randn(mp, R)
                W0 = W36[:, :R] + 0.15 * np.random.randn(pn, R) if R < 36 else W36 + 0.15 * np.random.randn(pn, R)
            else:
                U0 = np.random.randn(nm, R) * 0.3
                V0 = np.random.randn(mp, R) * 0.3
                W0 = np.random.randn(pn, R) * 0.3
            x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
            try:
                res = opt.least_squares(residual, x0, args=(R,), method='lm',
                                        max_nfev=1500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if res.cost < 1e-7:
                    x = res.x
                    sU = nm * R; sV = mp * R
                    Uo = x[:sU].reshape(nm, R); Vo = x[sU:sU+sV].reshape(mp, R); Wo = x[sU+sV:].reshape(pn, R)
                    Us, Vs, Ws = snap(Uo, Vo, Wo)
                    if verify(Us, Vs, Ws):
                        return (Us, Vs, Ws)
            except Exception:
                pass
        return None

    for R in range(35, 24, -1):
        if time_left() < 5:
            break
        sol = try_rank(R, 8)
        if sol is not None:
            return {"U": sol[0].tolist(), "V": sol[1].tolist(), "W": sol[2].tolist()}

    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
