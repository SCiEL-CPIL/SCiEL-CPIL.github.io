<<<<<<< SEARCH
def construct():
    t_start = time.time()
    deadline = 110.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    # clamp strictly inside to avoid invalid
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    # Multi-restart: alternate between local optimization from current best
    # and from slightly perturbed starts.
    restarts = 0
    while time.time() - t_start < deadline:
        restarts += 1
        # perturbation magnitude shrinks over restarts
        scale = 0.02 / (1.0 + 0.1 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        # ensure no coincident centres
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()
        # remaining time for this optimization
        rem = deadline - (time.time() - t_start)
        if rem < 5.0:
            break
        c_opt, r_opt = _optimize(c_start, max_iter=200000, rng=rng, deadline=rem - 3.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # final safety: ensure validity
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c
=======
def construct():
    t_start = time.time()
    deadline = 115.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    # Phase 1: Multi-restart gradient optimization
    restarts = 0
    while time.time() - t_start < deadline - 15.0:
        restarts += 1
        scale = 0.015 / (1.0 + 0.05 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()
        rem = deadline - 15.0 - (time.time() - t_start)
        if rem < 5.0:
            break
        c_opt, r_opt = _optimize(c_start, max_iter=100000, rng=rng, deadline=rem - 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # Phase 2: Final high-precision polish
    rem = deadline - (time.time() - t_start)
    if rem > 5.0:
        c_polished, r_polished = _local_polish(best_c, rem - 3.0)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # final safety: ensure validity
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c
>>>>>>> REPLACE
