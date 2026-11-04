<<<<<<< SEARCH
    # --- light local polish (nonnegativity enforced) -------------------------
    try:
        f0 = [max(1e-6, x) for x in best_f]
        fbest, _ = _nelder_mead(_R_smooth, f0, iters=400, step=0.15)
        for cand in (fbest, f0):
            cand = [max(0.0, x) for x in cand]
            v = _R(cand)
            if v > best:
                best, best_f = v, cand
    except Exception:
        pass
=======
    # --- DFT-parameterized gradient polish -----------------------------------
    try:
        f0 = np.array([max(1e-8, x) for x in best_f])
        f_dft = _dft_gradient_polish(f0, iters=150)
        f_dft = np.maximum(f_dft, 0.0)
        v_dft = _R(f_dft)
        if v_dft > best:
            best, best_f = v_dft, list(f_dft)
    except Exception:
        pass

    # --- light local polish (nonnegativity enforced) -------------------------
    try:
        f0 = [max(1e-6, x) for x in best_f]
        fbest, _ = _nelder_mead(_R_smooth, f0, iters=300, step=0.15)
        for cand in (fbest, f0):
            cand = [max(0.0, x) for x in cand]
            v = _R(cand)
            if v > best:
                best, best_f = v, cand
    except Exception:
        pass
>>>>>>> REPLACE
