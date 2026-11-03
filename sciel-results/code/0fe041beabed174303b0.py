import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    start_time = time.time()
    time_limit = 130  # wall-clock safety margin (evaluator kills at 150s)

    # Target tensor T[a*m+b, c*p+d, k*n+i] = 1 iff b==c and a==i and d==k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                T[i*m+b, b*p+k, k*n+i] = 1.0

    def snap_to_half_integers(U, V, W):
        return (np.round(U * 2) / 2.0,
                np.round(V * 2) / 2.0,
                np.round(W * 2) / 2.0)

    def max_residual(U, V, W):
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T))

    def verify(U, V, W, tol=1e-9):
        return max_residual(U, V, W) < tol

    # ---- Residual + analytic gradient (vectorized) ----
    def residual(x, R):
        sU = n*m*R
        sV = m*p*R
        U = x[0:sU].reshape(n*m, R)
        V = x[sU:sU+sV].reshape(m*p, R)
        W = x[sU+sV:].reshape(p*n, R)
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()

    def objective(x, R):
        r = residual(x, R)
        return 0.5 * float(np.dot(r, r))

    def gradient(x, R):
        sU = n*m*R
        sV = m*p*R
        U = x[0:sU].reshape(n*m, R)
        V = x[sU:sU+sV].reshape(m*p, R)
        W = x[sU+sV:].reshape(p*n, R)
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        res = T_rec - T
        gU = np.einsum('abc,bm,cm->am', res, V, W, optimize=True)
        gV = np.einsum('abc,am,cm->bm', res, U, W, optimize=True)
        gW = np.einsum('abc,am,bm->cm', res, U, V, optimize=True)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    # ---- Structured initializations ----
    def init_sparse_direct(R):
        """Each of the first min(R, n*m*p) terms targets one tensor entry directly."""
        U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
        cnt = 0
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    if cnt >= R:
                        break
                    U[i*m+b, cnt] = 1.0
                    V[b*p+k, cnt] = 1.0
                    W[k*n+i, cnt] = 1.0
                    cnt += 1
        return U, V, W

    def init_strassen_truncated(R):
        """Truncate a rank-36 Strassen-style basis to R columns."""
        U = np.zeros((n*m, 36)); V = np.zeros((m*p, 36)); W = np.zeros((p*n, 36))
        col = 0
        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0*m+j0, col]+=1; U[(i0+1)*m+(j0+1), col]+=1
            V[j0*p+k0, col]+=1; V[(j0+1)*p+(k0+1), col]+=1
            W[k0*n+i0, col]+=1; W[(k0+1)*n+(i0+1), col]+=1; col+=1
            U[(i0+1)*m+j0, col]+=1; U[(i0+1)*m+(j0+1), col]+=1
            V[j0*p+k0, col]+=1
            W[k0*n+(i0+1), col]+=1; W[(k0+1)*n+(i0+1), col]-=1; col+=1
            U[i0*m+j0, col]+=1
            V[j0*p+(k0+1), col]+=1; V[(j0+1)*p+(k0+1), col]-=1
            W[(k0+1)*n+i0, col]+=1; W[(k0+1)*n+(i0+1), col]+=1; col+=1
            U[(i0+1)*m+(j0+1), col]+=1
            V[(j0+1)*p+k0, col]+=1; V[j0*p+k0, col]-=1
            W[k0*n+i0, col]+=1; W[k0*n+(i0+1), col]+=1; col+=1
            U[i0*m+j0, col]+=1; U[i0*m+(j0+1), col]+=1
            V[(j0+1)*p+(k0+1), col]+=1
            W[k0*n+i0, col]-=1; W[(k0+1)*n+i0, col]+=1; col+=1
            U[(i0+1)*m+j0, col]+=1; U[i0*m+j0, col]-=1
            V[j0*p+k0, col]+=1; V[j0*p+(k0+1), col]+=1
            W[(k0+1)*n+(i0+1), col]+=1; col+=1
            U[i0*m+(j0+1), col]+=1; U[(i0+1)*m+(j0+1), col]-=1
            V[(j0+1)*p+k0, col]+=1; V[(j0+1)*p+(k0+1), col]+=1
            W[k0*n+i0, col]+=1; col+=1
        add_strassen(0,0,0); add_strassen(0,2,0)
        add_strassen(0,0,2); add_strassen(0,2,2)
        # fill remaining 28 with schoolbook 2x4x1 at k=4
        for di in range(2):
            for dj in range(4):
                U[(0+di)*m+(0+dj), col]=1
                V[(0+dj)*p+4, col]=1
                W[4*n+(0+di), col]=1
                col+=1
        return U[:, :R].copy(), V[:, :R].copy(), W[:, :R].copy()

    def init_random(R, scale, seed):
        rng = np.random.default_rng(seed)
        U = rng.normal(0, scale, (n*m, R))
        V = rng.normal(0, scale, (m*p, R))
        W = rng.normal(0, scale, (p*n, R))
        return U, V, W

    def build_seeds(R, seed_base):
        seeds = []
        seeds.append(('sparse', init_sparse_direct(R)))
        seeds.append(('strassen', init_strassen_truncated(R)))
        for s, scale in enumerate([1.0, 0.5, 2.0]):
            seeds.append((f'rand{s}', init_random(R, scale, seed_base + s)))
        return seeds

    # ---- Optimization driver with polish ----
    def optimize_from(U0, V0, W0, R, maxiter, maxfun, ftol, gtol, maxls):
        x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            res = opt.minimize(objective, x0, args=(R,), method='L-BFGS-B',
                               jac=gradient, maxiter=maxiter, maxfun=maxfun,
                               options={'ftol': ftol, 'gtol': gtol, 'maxls': maxls})
            x = res.x
        except Exception:
            return None
        sU = n*m*R; sV = m*p*R
        U = x[0:sU].reshape(n*m, R); V = x[sU:sU+sV].reshape(m*p, R); W = x[sU+sV:].reshape(p*n, R)
        return U, V, W

    def try_rank(R, seed_base):
        """Return (U,V,W) if an exact half-integer rank-R decomposition is found, else None."""
        seeds = build_seeds(R, seed_base)
        best = None
        for tag, (U0, V0, W0) in seeds:
            if time.time() - start_time > time_limit:
                break
            out = optimize_from(U0, V0, W0, R, maxiter=2500, maxfun=5000,
                                ftol=1e-12, gtol=1e-8, maxls=50)
            if out is None:
                continue
            U, V, W = out
            mr = max_residual(U, V, W)
            if mr < 1e-6:
                Us, Vs, Ws = snap_to_half_integers(U, V, W)
                if verify(Us, Vs, Ws, 1e-9):
                    return Us, Vs, Ws
            elif mr < 1e-3:
                # high-precision polish pass
                out2 = optimize_from(U, V, W, R, maxiter=8000, maxfun=16000,
                                     ftol=1e-15, gtol=1e-12, maxls=100)
                if out2 is not None:
                    U2, V2, W2 = out2
                    if max_residual(U2, V2, W2) < 1e-8:
                        Us, Vs, Ws = snap_to_half_integers(U2, V2, W2)
                        if verify(Us, Vs, Ws, 1e-9):
                            return Us, Vs, Ws
        return None

    # ---- Search from the trivial lower bound upward ----
    lower = 14  # ceil(40/3)
    for R in range(lower, 37):
        if time.time() - start_time > time_limit:
            break
        found = try_rank(R, seed_base=1000 + R)
        if found is not None:
            return {"U": found[0].tolist(), "V": found[1].tolist(), "W": found[2].tolist()}

    # ---- Fallback: rank-36 Strassen basis (always valid) ----
    U36, V36, W36 = init_strassen_truncated(36)
    Us, Vs, Ws = snap_to_half_integers(U36, V36, W36)
    if verify(Us, Vs, Ws, 1e-9):
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
