<<<<<<< SEARCH
def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - DEADLINE_MARGIN)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing (cheap, parallelizable)
        c_an = _repulsion_anneal(s, deadline, steps=2500)
        r_an = _compute_radius(c_an)

        # Stage B: smooth softmin polish
        c_pol, r_pol = _smooth_polish(c_an, deadline)

        r = max(r_an, r_pol)
        c = c_pol if r_pol >= r_an else c_an
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        c_pol, r_pol = _smooth_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c
=======
def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - DEADLINE_MARGIN)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing (cheap, parallelizable)
        c_an = _repulsion_anneal(s, deadline, steps=2000)
        r_an = _compute_radius(c_an)

        # Stage B: Lloyd/Voronoi equalization (smooth surrogate for min-pair)
        c_lloyd, r_lloyd = _lloyd_optimize(c_an, deadline, steps=20)
        r_lloyd = _compute_radius(c_lloyd)

        # Stage C: smooth softmin polish
        c_pol, r_pol = _smooth_polish(c_lloyd, deadline)

        # Stage D: targeted local search on the best so far from this seed
        candidates = [(c_an, r_an), (c_lloyd, r_lloyd), (c_pol, r_pol)]
        best_seed_c = max(candidates, key=lambda x: x[1])[0]
        c_ls, r_ls = _local_search(best_seed_c, deadline, max_iters=1000)

        r = max(r_an, r_lloyd, r_pol, r_ls)
        c = c_pol if r_pol == r else (c_lloyd if r_lloyd == r else (c_ls if r_ls == r else c_an))
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        # Try local search first (often more effective for final tweaks)
        c_ls, r_ls = _local_search(best_c, deadline, max_iters=500)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls
        # Then smooth polish
        c_pol, r_pol = _smooth_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c
>>>>>>> REPLACE
