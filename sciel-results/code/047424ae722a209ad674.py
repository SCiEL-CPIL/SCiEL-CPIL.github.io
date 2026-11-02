<<<<<<< SEARCH
    if best_config is None:
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    r_final = r * (1 - 1e-6)
    return np.column_stack([xs, ys, np.full(N, r_final)])
=======
    if best_config is None:
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    r_final = r * (1 - 1e-6)
    return np.column_stack([xs, ys, np.full(N, r_final)])
>>>>>>> REPLACE
