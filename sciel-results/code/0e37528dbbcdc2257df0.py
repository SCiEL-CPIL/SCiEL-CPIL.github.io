<<<<<<< SEARCH
    # ---- Refinement passes, each guarded by time, each may improve best ----
    def _try_update(res):
        nonlocal best_cfg, best_score, best_center, best_angle
        if res is None:
            return
        rcfg, rcenter, rangle, rs = res
        if rs is not None and rs < best_score:
            best_cfg = [row[:] for row in rcfg]
            best_center = rcenter[:]
            best_angle = rangle
            best_score = rs

    # L-BFGS-B smooth surrogate polish
    if time.time() - t0 < TIME_LIMIT - 20.0:
        _try_update(_lbfgs_polish(best_cfg, best_center, best_angle,
                                  min(20.0, TIME_LIMIT - (time.time()-t0) - 20.0)))

    # Nelder-Mead polish (now actually used)
    if time.time() - t0 < TIME_LIMIT - 10.0:
        _try_update(_nelder_mead_polish(best_cfg, best_center, best_angle,
                                        min(12.0, TIME_LIMIT - (time.time()-t0) - 10.0)))

    # One more L-BFGS-B pass from the NM result (different basin)
    if time.time() - t0 < TIME_LIMIT - 8.0:
        _try_update(_lbfgs_polish(best_cfg, best_center, best_angle,
                                  min(8.0, TIME_LIMIT - (time.time()-t0) - 8.0), sigma=0.03))

    # Final outer-hex grid search (inner fixed, already valid)
    if time.time() - t0 < TIME_LIMIT - 4.0:
        gs_s, gs_cx, gs_cy, gs_ang = _grid_search_outer(
            best_cfg, best_center[0], best_center[1], best_angle,
            TIME_LIMIT - (time.time()-t0) - 4.0)
        if gs_s < best_score:
            best_score = gs_s
            best_center = [gs_cx, gs_cy]
            best_angle = gs_ang
=======
    # ---- Refinement passes, each guarded by time, each may improve best ----
    def _try_update(res):
        nonlocal best_cfg, best_score, best_center, best_angle
        if res is None:
            return
        rcfg, rcenter, rangle, rs = res
        if rs is not None and rs < best_score:
            best_cfg = [row[:] for row in rcfg]
            best_center = rcenter[:]
            best_angle = rangle
            best_score = rs

    # 1. Initial L-BFGS-B polish with moderate sigma
    if time.time() - t0 < TIME_LIMIT - 40.0:
        budget = min(30.0, TIME_LIMIT - (time.time()-t0) - 40.0)
        _try_update(_lbfgs_polish(best_cfg, best_center, best_angle, budget, sigma=0.08))

    # 2. Nelder-Mead polish to escape local minima of the smooth surrogate
    if time.time() - t0 < TIME_LIMIT - 20.0:
        budget = min(15.0, TIME_LIMIT - (time.time()-t0) - 20.0)
        _try_update(_nelder_mead_polish(best_cfg, best_center, best_angle, budget))

    # 3. Second L-BFGS-B pass with tighter sigma for high precision
    if time.time() - t0 < TIME_LIMIT - 10.0:
        budget = min(15.0, TIME_LIMIT - (time.time()-t0) - 10.0)
        _try_update(_lbfgs_polish(best_cfg, best_center, best_angle, budget, sigma=0.04))

    # 4. Final outer-hex grid search (inner fixed, already valid)
    if time.time() - t0 < TIME_LIMIT - 4.0:
        gs_s, gs_cx, gs_cy, gs_ang = _grid_search_outer(
            best_cfg, best_center[0], best_center[1], best_angle,
            TIME_LIMIT - (time.time()-t0) - 4.0)
        if gs_s < best_score:
            best_score = gs_s
            best_center = [gs_cx, gs_cy]
            best_angle = gs_ang
>>>>>>> REPLACE
