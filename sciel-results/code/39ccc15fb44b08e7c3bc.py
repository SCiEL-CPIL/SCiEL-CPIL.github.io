    def residual_func(x):
        U = x[:nm * r].reshape(nm, r)
        V = x[nm * r:nm * r + mp * r].reshape(mp, r)
        W = x[nm * r + mp * r:].reshape(pn, r)
        return _bilinear_residuals(U, V, W, n, m, p)
    
    try:
        from scipy.optimize import least_squares
        res = least_squares(residual_func, x0, method='trf', max_nfev=200)
        if res.cost < 1e-10:
            # Verify
            x = res.x
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_full(U, V, W, n, m, p, tol=1e-5):
                return U.tolist(), V.tolist(), W.tolist()
    except Exception:
        pass
    return None
