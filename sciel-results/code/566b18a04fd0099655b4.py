<<<<<<< SEARCH
def _try_rank(n, m, p, r, n_trials=3):
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]

    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT:
            return None
        rng = np.random.RandomState(t * 7919 + r)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)

        for it in range(500):
            if time.time() - _START > _TIME_LIMIT:
                return None
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv

            if loss < 1e-14:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if _verify(n, m, p, U, V, W):
                    return (U, V, W)

            lr = 0.02 / (1.0 + it * 0.002)
            U -= lr * Ug
            V -= lr * Vg
            W -= lr * Wg

    return None
=======
def _try_rank(n, m, p, r, n_trials=5):
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]

    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT - 5:
            return None
        rng = np.random.RandomState(t * 7919 + r)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)

        # Custom gradient descent with adaptive learning rate
        for it in range(1000):
            if time.time() - _START > _TIME_LIMIT - 5:
                return None
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv

            if loss < 1e-10:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(n, m, p, U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                # Also try integer rounding
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if _verify(n, m, p, U_i, V_i, W_i):
                    return (U_i, V_i, W_i)

            # Adaptive learning rate with momentum
            lr = 0.05 / (1.0 + it * 0.005)
            U -= lr * Ug
            V -= lr * Vg
            W -= lr * Wg

        # If gradient descent didn't find exact solution, try scipy minimize
        if time.time() - _START > _TIME_LIMIT - 5:
            return None
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            if time.time() - _START > _TIME_LIMIT - 5:
                return 1e10
            U_x = x[:nm*r].reshape(nm, r)
            V_x = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W_x = x[nm*r+mp*r:].reshape(pn, r)
            loss = 0.0
            for ui, vi, wi in idx:
                s = np.dot(U_x[ui], V_x[vi] * W_x[wi])
                loss += (s - 1.0) ** 2
            return loss

        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e-8:
                x_opt = res.x
                U_opt = x_opt[:nm*r].reshape(nm, r)
                V_opt = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
                W_opt = x_opt[nm*r+mp*r:].reshape(pn, r)
                # Try rounding
                for scale in [2, 1]:
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    if _verify(n, m, p, U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
        except Exception:
            pass

    return None
>>>>>>> REPLACE
