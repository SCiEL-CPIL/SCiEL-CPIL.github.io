<<<<<<< SEARCH
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    w_final, pos_final = best_result
    r_final = _compute_radius(pos_final, w_final)

    # Build output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = pos_final[:, 0]
    result[:, 1] = pos_final[:, 1]
    result[:, 2] = r_final

    return result
=======
    # Build output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = pos_final[:, 0]
    result[:, 1] = pos_final[:, 1]
    result[:, 2] = r_final

    return result
>>>>>>> REPLACE
