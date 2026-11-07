    if norm_best is not None and norm_best <= 1e-8:
        return x_best
    # ... restarts ...
    if x_best is not None:
        return x_best
    return np.array(x0, dtype=float)
