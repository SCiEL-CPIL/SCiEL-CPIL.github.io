<<<<<<< SEARCH
    # ---- Phase 2: full 400-dim polish with smooth surrogate (FFT gradient) ----
    h_curr = best_h.copy()
    # normalize sum to 1 to keep objective well-scaled
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0]
    for beta in beta_seq:
        if time_left() < 25.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 6000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass
=======
    # ---- Phase 2: full 400-dim polish with smooth surrogate (FFT gradient) ----
    h_curr = best_h.copy()
    # normalize sum to 1 to keep objective well-scaled
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    for beta in beta_seq:
        if time_left() < 20.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 8000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass
>>>>>>> REPLACE

