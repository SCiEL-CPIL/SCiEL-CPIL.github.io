def _try_rank27(T_flat, U28, V28, W28, deadline):
    from scipy.optimize import least_squares
    R = 27
    best = None
    for drop_idx in range(28):
        if time.time() > deadline - 3: break
        keep = [i for i in range(28) if i != drop_idx]
        U = U28[:, keep].astype(float)
        V = V28[:, keep].astype(float)
        W = W28[:, keep].astype(float)
        x0 = _pack(U, V, W)
        # a couple of perturbations
        rng = np.random.default_rng(drop_idx)
        attempts = [x0]
        for _ in range(2):
            attempts.append(x0 + rng.normal(0, 0.05, x0.size))
        for x0a in attempts:
            if time.time() > deadline - 3: break
            def fun(x): 
                U,V,W=_unpack(x); return T_flat - np.einsum('ar,br,cr->abc',U,V,W).ravel()
            def jac(x):
                U,V,W=_unpack(x); return _build_J(U,V,W,R)
            try:
                sol = least_squares(fun, x0a, jac=jac, method='trf',
                                    xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                    max_nfev=120)
            except Exception:
                continue
            U,V,W = _unpack(sol.x)
            # round to half-integers
            Ur=_round_half(U); Vr=_round_half(V); Wr=_round_half(W)
            recon = np.einsum('ar,br,cr->abc', Ur,Vr,Wr)
            if np.max(np.abs(T_flat - recon.ravel())) < 1e-8:
                return Ur.tolist(), Vr.tolist(), Wr.tolist()
            # also check unrounded exactness (in case evaluator is lenient) - but we require half-int, so skip
    return None
