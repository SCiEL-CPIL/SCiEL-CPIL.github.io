<<<<<<< SEARCH
def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    best = (inner, outer_c, s, rot)
    best_pen = 1.0 if not _valid(inner, outer_c, s, rot) else 0.0
    cur = (inner, outer_c, s, rot)
    cur_pen = best_pen
    
    # Tighten s first
    s_tight = _min_outer_side(cur[0], cur[1], cur[3])
    if s_tight < cur[2]:
        cur = (cur[0], cur[1], s_tight, cur[3])
        if _valid(cur[0], cur[1], cur[2], cur[3]):
            best = cur
            best_pen = 0.0
            cur_pen = 0.0
    
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        # Perturb
        cand = [[x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), a + rng.uniform(-15, 15)] for (x, y, a) in cur[0]]
        cand_oc = (cur[1][0] + rng.uniform(-0.2, 0.2), cur[1][1] + rng.uniform(-0.2, 0.2))
        cand_rot = cur[3] + rng.uniform(-10, 10)
        
        # Compute tight s for this candidate
        cand_s = _min_outer_side(cand, cand_oc, cand_rot)
        
        if _valid(cand, cand_oc, cand_s, cand_rot):
            cur = (cand, cand_oc, cand_s, cand_rot)
            cur_pen = 0.0
            if cand_s < best[2]:
                best = cur
                best_pen = 0.0
        else:
            pass
    
    return best
=======
def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    """
    Use differential_evolution to optimize inner hexagon positions.
    """
    # Convert inner to parameter vector
    params0 = []
    for (x, y, a) in inner:
        params0.extend([x, y, a])
    params0 = np.array(params0)
    
    # Define bounds: positions in [-6, 6], angles in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-6.0, 6.0))  # x
        bounds.append((-6.0, 6.0))  # y
        bounds.append((0.0, 60.0))  # angle
    
    # Time budget for DE
    de_budget = min(30.0, budget - time.time())
    if de_budget < 5.0:
        # Not enough time, just return current
        return (inner, outer_c, s, rot)
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            args=(outer_c, rot),
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=int(rng.random() * 10000),
            polish=True,
            init='sobol'
        )
        
        if result.success or result.fun < s:
            best_params = result.x
            inner_opt = []
            for i in range(N):
                inner_opt.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
            
            s_opt = _min_outer_side(inner_opt, outer_c, rot)
            if _valid(inner_opt, outer_c, s_opt, rot):
                return (inner_opt, outer_c, s_opt, rot)
    except Exception:
        pass
    
    # Fallback: return current
    return (inner, outer_c, s, rot)
>>>>>>> REPLACE

