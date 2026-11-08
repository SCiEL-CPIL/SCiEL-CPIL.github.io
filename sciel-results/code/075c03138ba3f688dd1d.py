    # ---- Refine with Nelder-Mead on DRAG parameters ----
    def opt_drag(p):
        if time.time() > deadline - 40:
            return 1e6
        I_Q = make_drag_pulse(p)
        if I_Q is None:
            return 1e6
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    try:
        r = minimize(opt_drag, best_params, method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 0.001, 'fatol': 1e-6})
        if r.fun < best_s:
            I_Q = make_drag_pulse(r.x)
            if I_Q is not None and check_constraints(I_Q):
                best_s, best = r.fun, I_Q
    except Exception:
        pass

    if not check_constraints(best):
        best = seed_pulse()
    return best
