    if not _is_valid(best_inner):
        # Fallback to sparse valid arrangement
        best_inner = [[i*10.0, 0.0, 0.0] for i in range(12)]
        best_center = (0.0, 0.0)
        best_side = _outer_hex_side(best_inner, best_center, 0.0)
        best_angle = 0.0
