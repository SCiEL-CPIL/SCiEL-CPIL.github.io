<<<<<<< SEARCH
def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s kill

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _raw_value(best_h)
    best_score = base_score

    # Generate diverse initializations
    inits = []

    inits.append(np.ones(N) / N)

    h_chirp = _gen_chirp(N)
    inits.append(_normalize_sum(h_chirp))

    for seed in [1, 7, 13, 42, 99]:
        h_rand = _gen_random(N, seed)
        inits.append(_normalize_sum(h_rand))

    for wtype in ['gaussian', 'hamming', 'triangle', 'blackman']:
        h_win = _gen_window(N, wtype)
        inits.append(_normalize_sum(h_win))

    for seed in [5, 11, 23, 37, 53]:
        h_smooth = _gen_smoothed_random(N, seed)
        inits.append(_normalize_sum(h_smooth))

    for seed in [3, 17]:
        rng = np.random.default_rng(seed)
        env = np.exp(-((np.arange(N) - N / 2)**2) / (2 * (N / 4)**2))
        signs = rng.choice([-1, 1], size=N)
        h_alt = signs * env
        inits.append(_normalize_sum(h_alt))

    for freq_combo in [(1, 2), (1, 3), (2, 5), (3, 7)]:
        f1, f2 = freq_combo
        i = np.arange(N)
        h_multi = np.cos(2 * np.pi * f1 * i / N) + 0.5 * np.cos(2 * np.pi * f2 * i / N)
        inits.append(_normalize_sum(h_multi))

    # Optimization pipeline for each init
    for h_init in inits:
        if time.time() > deadline - 25:
            break

        h_work = h_init.copy()

        # Stage 1: Spectral subspace optimization
        remaining = deadline - time.time()
        if remaining > 20:
            h_spec, val_spec = _spectral_subspace_optimize(h_work, dim=8,
                                                           deadline=deadline - 15)
            h_work = h_spec

        # Stage 2: L-BFGS smooth optimization
        remaining = deadline - time.time()
        if remaining > 15:
            h_opt, val_opt = _lbfgs_optimize(h_work, beta=50.0, max_iter=200,
                                             deadline=deadline - 10)
            h_work = h_opt

        # Stage 3: Block sign-flip local search
        remaining = deadline - time.time()
        if remaining > 10:
            h_block, val_block = _block_sign_flip_local_search(h_work,
                                                               block_sizes=[1, 2, 4, 8, 16, 32],
                                                               max_passes=4,
                                                               deadline=deadline - 5)
            h_work = h_block

        # Stage 4: Single flip local search (incremental, fast)
        remaining = deadline - time.time()
        if remaining > 5:
            h_flip, val_flip = _single_flip_local_search(h_work, max_passes=12,
                                                         deadline=deadline - 3)
            h_work = h_flip

        val = _raw_value(h_work)
        if val < best_score:
            best_score = val
            best_h = h_work.copy()

    # Final refinement on the best solution
    remaining = deadline - time.time()
    if remaining > 20:
        h_final, val_final = _block_sign_flip_local_search(best_h,
                                                           block_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
                                                           max_passes=6,
                                                           deadline=deadline - 15)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final

    remaining = deadline - time.time()
    if remaining > 12:
        h_final2, val_final2 = _single_flip_local_search(best_h, max_passes=20,
                                                         deadline=deadline - 8)
        if val_final2 < best_score:
            best_score = val_final2
            best_h = h_final2

    # Epigraph polish (high-precision final pass)
    remaining = deadline - time.time()
    if remaining > 10:
        h_epi, val_epi = _epigraph_polish(best_h, deadline=deadline - 2)
        if val_epi < best_score:
            best_score = val_epi
            best_h = h_epi

    # Ensure we return something valid
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        best_score = _raw_value(best_h)

    return best_h
=======
def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _raw_value(best_h)
    best_score = base_score

    # Generate diverse initializations
    inits = []

    inits.append(np.ones(N) / N)

    h_chirp = _gen_chirp(N)
    inits.append(_normalize_sum(h_chirp))

    for seed in [1, 7, 13, 42, 99]:
        h_rand = _gen_random(N, seed)
        inits.append(_normalize_sum(h_rand))

    for wtype in ['gaussian', 'hamming', 'triangle', 'blackman']:
        h_win = _gen_window(N, wtype)
        inits.append(_normalize_sum(h_win))

    for seed in [5, 11, 23, 37, 53]:
        h_smooth = _gen_smoothed_random(N, seed)
        inits.append(_normalize_sum(h_smooth))

    for seed in [3, 17]:
        rng = np.random.default_rng(seed)
        env = np.exp(-((np.arange(N) - N / 2)**2) / (2 * (N / 4)**2))
        signs = rng.choice([-1, 1], size=N)
        h_alt = signs * env
        inits.append(_normalize_sum(h_alt))

    for freq_combo in [(1, 2), (1, 3), (2, 5), (3, 7)]:
        f1, f2 = freq_combo
        i = np.arange(N)
        h_multi = np.cos(2 * np.pi * f1 * i / N) + 0.5 * np.cos(2 * np.pi * f2 * i / N)
        inits.append(_normalize_sum(h_multi))

    # Phase 1: Fast multi-start with cheap L-BFGS to identify top candidates
    fast_results = []
    for h_init in inits:
        if time.time() > deadline - 100:
            break
        h_work = h_init.copy()
        remaining = deadline - time.time()
        if remaining > 8:
            h_opt, val_opt = _lbfgs_optimize(h_work, beta=50.0, max_iter=80,
                                             deadline=deadline - 5)
            h_work = h_opt
        val = _raw_value(h_work)
        fast_results.append((val, h_work))

    # Sort by score, take top 5 for expensive pipeline
    fast_results.sort(key=lambda x: x[0])
    top_candidates = [h for _, h in fast_results[:5]]

    # Phase 2: Expensive pipeline on top candidates
    for h_init in top_candidates:
        if time.time() > deadline - 30:
            break

        h_work = h_init.copy()

        # Stage 1: Spectral subspace optimization
        remaining = deadline - time.time()
        if remaining > 25:
            h_spec, val_spec = _spectral_subspace_optimize(h_work, dim=8,
                                                           deadline=deadline - 20)
            h_work = h_spec

        # Stage 2: L-BFGS smooth optimization
        remaining = deadline - time.time()
        if remaining > 20:
            h_opt, val_opt = _lbfgs_optimize(h_work, beta=50.0, max_iter=200,
                                             deadline=deadline - 15)
            h_work = h_opt

        # Stage 3: Block sign-flip local search
        remaining = deadline - time.time()
        if remaining > 12:
            h_block, val_block = _block_sign_flip_local_search(h_work,
                                                               block_sizes=[1, 2, 4, 8, 16, 32],
                                                               max_passes=4,
                                                               deadline=deadline - 8)
            h_work = h_block

        # Stage 4: Single flip local search (vectorized, fast)
        remaining = deadline - time.time()
        if remaining > 8:
            h_flip, val_flip = _single_flip_local_search(h_work, max_passes=15,
                                                         deadline=deadline - 5)
            h_work = h_flip

        val = _raw_value(h_work)
        if val < best_score:
            best_score = val
            best_h = h_work.copy()

    # Final refinement on the best solution
    remaining = deadline - time.time()
    if remaining > 20:
        h_final, val_final = _block_sign_flip_local_search(best_h,
                                                           block_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
                                                           max_passes=6,
                                                           deadline=deadline - 15)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final

    remaining = deadline - time.time()
    if remaining > 15:
        h_final2, val_final2 = _single_flip_local_search(best_h, max_passes=25,
                                                         deadline=deadline - 10)
        if val_final2 < best_score:
            best_score = val_final2
            best_h = h_final2

    # Epigraph polish (high-precision final pass)
    remaining = deadline - time.time()
    if remaining > 12:
        h_epi, val_epi = _epigraph_polish(best_h, deadline=deadline - 3)
        if val_epi < best_score:
            best_score = val_epi
            best_h = h_epi

    # Final single-flip polish
    remaining = deadline - time.time()
    if remaining > 5:
        h_final3, val_final3 = _single_flip_local_search(best_h, max_passes=10,
                                                         deadline=deadline - 1)
        if val_final3 < best_score:
            best_score = val_final3
            best_h = h_final3

    # Ensure we return something valid
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        best_score = _raw_value(best_h)

    return best_h
>>>>>>> REPLACE
