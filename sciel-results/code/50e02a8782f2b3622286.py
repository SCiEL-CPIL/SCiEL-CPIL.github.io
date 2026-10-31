<<<<<<< SEARCH
    # Final high-precision polish
    if best_cx is not None and time.time() < deadline - 10:
        cx_p, cy_p, r_p = _high_precision_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 5)
        if _is_valid(cx_p, cy_p, r_p, best_W, best_H):
            val_p = np.sum(r_p)
            if val_p > best_value:
                best_value = val_p
                best_cx, best_cy, best_r = cx_p, cy_p, r_p
=======
    # Final high-precision polish
    if best_cx is not None and time.time() < deadline - 10:
        cx_p, cy_p, r_p = _high_precision_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 3)
        if _is_valid(cx_p, cy_p, r_p, best_W, best_H):
            val_p = np.sum(r_p)
            if val_p > best_value:
                best_value = val_p
                best_cx, best_cy, best_r = cx_p, cy_p, r_p
>>>>>>> REPLACE
