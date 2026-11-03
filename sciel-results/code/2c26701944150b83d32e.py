<<<<<<< SEARCH
def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Find best outer hexagon angle and center via LP.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    coarse = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_angle >= 0:
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.01)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

        if time.time() < start + time_limit:
            uf_lo = max(0.0, best_angle - 0.05)
            uf_hi = min(30.0, best_angle + 0.05)
            ultra_fine = np.arange(uf_lo, uf_hi + 0.001, 0.001)
            for angle_deg in ultra_fine:
                if time.time() > start + time_limit:
                    break
                res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle
=======
def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Find best outer hexagon angle and center via LP.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse scan: 0 to 30 degrees (hexagon symmetry is 60 deg, but we only need 30
    # because rotating the outer hex by 30 deg is equivalent to flipping its orientation)
    coarse = np.arange(0, 30.01, 1.0)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_angle >= 0:
        # Fine scan around best
        fine_lo = max(0.0, best_angle - 1.0)
        fine_hi = min(30.0, best_angle + 1.0)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.05)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

        # Ultra-fine
        if time.time() < start + time_limit:
            uf_lo = max(0.0, best_angle - 0.1)
            uf_hi = min(30.0, best_angle + 0.1)
            ultra_fine = np.arange(uf_lo, uf_hi + 0.001, 0.002)
            for angle_deg in ultra_fine:
                if time.time() > start + time_limit:
                    break
                res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        best_angle = float(angle_deg)

        # Final ultra-precise
        if time.time() < start + time_limit:
            fp_lo = max(0.0, best_angle - 0.01)
            fp_hi = min(30.0, best_angle + 0.01)
            final_pts = np.arange(fp_lo, fp_hi + 0.001, 0.0001)
            for angle_deg in final_pts:
                if time.time() > start + time_limit:
                    break
                res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle
>>>>>>> REPLACE

