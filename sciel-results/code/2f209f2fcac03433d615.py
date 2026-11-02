<<<<<<< SEARCH
    # ---------- Final high-precision polish on best result ----------
    if best_result is not None and time.time() < deadline - 10.0:
=======
    # ---------- Decoupled Optimization on best result ----------
    if best_result is not None and time.time() < deadline - 15.0:
        xs_b = best_result[:, 0].copy()
        ys_b = best_result[:, 1].copy()
        rs_b = best_result[:, 2].copy()
        min_x = np.min(xs_b - rs_b)
        max_x = np.max(xs_b + rs_b)
        min_y = np.min(ys_b - rs_b)
        max_y = np.max(ys_b + rs_b)
        w_b = max_x - min_x
        h_b = max_y - min_y
        xs_b = xs_b - min_x
        ys_b = ys_b - min_y
        
        w_dec, h_dec, xs_dec, ys_dec, rs_dec = _decoupled_optimize(w_b, h_b, xs_b, ys_b, deadline)
        if np.any(rs_dec > 0):
            res_dec = _make_result_per_circle(w_dec, h_dec, xs_dec, ys_dec, rs_dec)
            consider(res_dec)

    # ---------- Final high-precision polish on best result ----------
    if best_result is not None and time.time() < deadline - 10.0:
>>>>>>> REPLACE
