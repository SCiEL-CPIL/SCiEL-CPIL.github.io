<<<<<<< SEARCH
def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    dist = _init_dist(pts)
    best = pts.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)

    # Phase 1: Multi-start annealing within the time budget
    while time.time() - t0 < DEADLINE - 20.0:  # reserve 20s for polishing
        # small perturbation of the seed for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)
        
        cand_dist = _init_dist(cand)

        t_left = DEADLINE - 20.0 - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 15.0)  # cap per-anneal budget
        cand, cand_dist, r = _anneal(cand, cand_dist, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
            best_dist = cand_dist

    # Phase 2: High-precision polish pass
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 5.0:
        best, best_dist, best_r = _polish(best, best_dist, t_left - 2.0, rng)

    # final safety: ensure valid output
    final_r = _score_fast(best, best_dist)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
=======
def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Try multiple initializations
    inits = []
    
    # Hex lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts_hex = _lattice_seed(spacing)
    inits.append(pts_hex)
    
    # Radial rings
    pts_ring = _radial_ring_init()
    inits.append(pts_ring)
    
    # Slightly perturbed hex
    jitter = 0.02 * rng.standard_normal((N, 2))
    pts_hex2 = pts_hex + jitter
    rad = np.linalg.norm(pts_hex2, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        pts_hex2[mask] = pts_hex2[mask] / rad[mask, None] * (1.0 - 1e-6)
    inits.append(pts_hex2)
    
    # Best initial
    best = None
    best_r = 0.0
    for init_pts in inits:
        init_dist = _init_dist(init_pts)
        r = _score_fast(init_pts, init_dist)
        if r > best_r:
            best_r = r
            best = init_pts.copy()
            best_dist = init_dist.copy()
    
    # Phase 1: Multi-start annealing within the time budget
    # Reserve time for polishing
    polish_reserve = 30.0
    anneal_deadline = DEADLINE - polish_reserve
    
    n_anneals = 0
    while time.time() - t0 < anneal_deadline:
        # small perturbation of the current best for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)
        
        cand_dist = _init_dist(cand)

        t_left = anneal_deadline - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 12.0)  # cap per-anneal budget
        cand, cand_dist, r = _anneal(cand, cand_dist, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
            best_dist = cand_dist
        n_anneals += 1
        
        # After first few anneals, use smaller perturbations
        if n_anneals > 3:
            jitter_scale = 0.005
        elif n_anneals > 1:
            jitter_scale = 0.008
        else:
            jitter_scale = 0.01
    
    # Phase 2: High-precision polish pass with L-BFGS
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 10.0:
        best, best_dist, best_r = _polish(best, best_dist, t_left - 3.0, rng)

    # final safety: ensure valid output
    final_r = _score_fast(best, best_dist)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))
        best_dist = _init_dist(best)

    return best
>>>>>>> REPLACE
