    final_penalty = calc_overlap_penalty(hexes)
    if final_penalty > 1e-6:
        # Return seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
