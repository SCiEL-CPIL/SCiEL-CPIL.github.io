<<<<<<< SEARCH
    # Extra SLSQP polish if time permits
    if time.time() < deadline - 3:
        try:
            pos_extra, rad_extra = _slsqp_polish(best_pos, best_radii, deadline, maxiter=200)
            pos_pp, radii_pp = _post_process(pos_extra, rad_extra)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
=======
    # Extra SLSQP polish if time permits
    if time.time() < deadline - 5:
        try:
            pos_extra, rad_extra = _slsqp_polish(best_pos, best_radii, deadline, maxiter=300)
            pos_pp, radii_pp = _post_process(pos_extra, rad_extra)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final QP radius optimization
    if time.time() < deadline - 2:
        try:
            radii_opt = _qp_radius_solver(best_pos, best_radii)
            pos_pp, radii_pp = _post_process(best_pos, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
>>>>>>> REPLACE
