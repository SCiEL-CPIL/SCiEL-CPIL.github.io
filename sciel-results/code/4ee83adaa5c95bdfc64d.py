<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci_points(N, seed=42, jitter=0.015),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
        _jittered_fibonacci_points(N, seed=123, jitter=0.03),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Repulsion-based refinement on each initial config
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_per = min(remaining / (len(initial_configs) - i), 10.0)
        rep_pts, rep_cr = _repulsion_refine(init_pts, time_limit=time_per, start_time=start_time)
        if rep_cr < best_cr:
            best_cr = rep_cr
            best_points = rep_pts.copy()

    # Phase 2: Full multi-stage refinement on best
    remaining = deadline - time.time()
    if remaining > 10:
        refine_time = min(remaining * 0.45, 35.0)
        refined, cr = _refine(best_points, time_limit=refine_time, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 3: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 12:
        sa_time = min(remaining * 0.2, 12.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 4: Final refinement
    remaining = deadline - time.time()
    if remaining > 8:
        fin_time = min(remaining * 0.4, 18.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Phase 5: Final L-BFGS polish with very high beta
    remaining = deadline - time.time()
    if remaining > 5:
        lbfgs_time = max(remaining * 0.6, 3.0)
        lbfgs_pts, lbfgs_cr = _lbfgs_refine(best_points, max_time=lbfgs_time,
                                            start_time=start_time, beta=300.0)
        if lbfgs_cr < best_cr:
            best_cr = lbfgs_cr
            best_points = lbfgs_pts.copy()

    # Phase 6: Riesz-assisted final polish
    remaining = deadline - time.time()
    if remaining > 4:
        riesz_time = max(remaining * 0.4, 2.0)
        r_pts, r_cr = _lbfgs_riesz(best_points, max_time=riesz_time,
                                    start_time=start_time, s=2.5, weight=0.02)
        if r_cr < best_cr:
            best_cr = r_cr
            best_points = r_pts.copy()

    # Phase 7: Final GD micro-polish
    remaining = deadline - time.time()
    if remaining > 2:
        gd_time = max(remaining * 0.7, 1.0)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=500, lr_init=0.002,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = _normalize(best_points)
    return best_points
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci_points(N, seed=42, jitter=0.015),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
        _jittered_fibonacci_points(N, seed=123, jitter=0.03),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Quick repulsion refinement on each initial config
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_per = min(remaining / (len(initial_configs) - i), 8.0)
        rep_pts, rep_cr = _repulsion_refine(init_pts, time_limit=time_per, start_time=start_time)
        if rep_cr < best_cr:
            best_cr = rep_cr
            best_points = rep_pts.copy()

    # Phase 2: Full multi-stage refinement on best
    remaining = deadline - time.time()
    if remaining > 12:
        refine_time = min(remaining * 0.40, 30.0)
        refined, cr = _refine(best_points, time_limit=refine_time, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 3: Simulated annealing for basin hopping
    remaining = deadline - time.time()
    if remaining > 12:
        sa_time = min(remaining * 0.18, 10.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 4: Second full refinement after SA
    remaining = deadline - time.time()
    if remaining > 10:
        fin_time = min(remaining * 0.35, 20.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Phase 5: Multi-beta LSE final polish
    remaining = deadline - time.time()
    if remaining > 8:
        mb_time = min(remaining * 0.3, 12.0)
        mb_pts, mb_cr = _multi_beta_lse(best_points, time_limit=mb_time, start_time=start_time)
        if mb_cr < best_cr:
            best_cr = mb_cr
            best_points = mb_pts.copy()

    # Phase 6: Exact local solve final polish
    remaining = deadline - time.time()
    if remaining > 5:
        exact_time = min(remaining * 0.3, 6.0)
        ex_pts, ex_cr = _exact_local_solve(best_points, time_limit=exact_time,
                                           start_time=start_time)
        if ex_cr < best_cr:
            best_cr = ex_cr
            best_points = ex_pts.copy()

    # Phase 7: Final GD micro-polish
    remaining = deadline - time.time()
    if remaining > 2:
        gd_time = max(remaining * 0.6, 1.0)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=500, lr_init=0.002,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = _normalize(best_points)
    return best_points
>>>>>>> REPLACE
