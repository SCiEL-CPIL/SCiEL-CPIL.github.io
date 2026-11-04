<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # Phase 1: Global Search with Splines
    if time.time() < deadline - 60.0:
        n_pts = 12
        bounds = [(0.0, 5.0)] * n_pts
        try:
            res = differential_evolution(
                objective_spline, bounds, maxiter=100, popsize=20,
                tol=1e-6, seed=42, polish=True, init='sobol',
                maxfevals=3000
            )
            h_opt = spline_to_h(res.x, n_pts)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 2: Global Search with Gaussians (more components)
    if time.time() < deadline - 40.0:
        for n_gauss in [5, 7]:
            if time.time() > deadline - 30.0:
                break
            n_params = 3 * n_gauss + 1
            bounds = []
            for _ in range(n_gauss):
                bounds.append((-0.25, 0.25))
                bounds.append((0.01, 0.15))
                bounds.append((0.0, 5.0))
            bounds.append((0.0, 2.0))
            
            try:
                res = differential_evolution(
                    objective_gauss, bounds, maxiter=80, popsize=20,
                    tol=1e-6, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = gaussians_to_h(res.x)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass
                
    # Phase 2.5: Global Search with Blocks
    if time.time() < deadline - 30.0:
        for n_blocks in [10, 20, 50]:
            if time.time() > deadline - 20.0:
                break
            bounds = [(0.0, 5.0)] * n_blocks
            try:
                res = differential_evolution(
                    lambda p: objective_block(p, n_blocks), bounds, maxiter=100, popsize=20,
                    tol=1e-6, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = blocks_to_h(res.x, n_blocks)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass

    # Phase 3: Local Refinement on full 600-dim space
    # Start from best global solution and some random starts
    starts = [best_h]
    for _ in range(5):
        # Random smooth start
        h_rand = np.abs(rng.normal(1.0, 0.5, N))
        # Smooth it
        kernel = np.ones(10) / 10
        h_rand = np.convolve(h_rand, kernel, mode='same')
        starts.append(h_rand)
        
    for h_start in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            res = opt.minimize(
                obj_smooth, h_start, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 4: Final Polish
    if time.time() < deadline - 2.0:
        try:
            res = opt.minimize(
                obj_smooth, best_h, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    return list(best_h)
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # Phase 1: Global Search with Splines
    if time.time() < deadline - 60.0:
        n_pts = 12
        bounds = [(0.0, 5.0)] * n_pts
        try:
            res = differential_evolution(
                objective_spline, bounds, maxiter=100, popsize=20,
                tol=1e-6, seed=42, polish=True, init='sobol',
                maxfevals=3000
            )
            h_opt = spline_to_h(res.x, n_pts)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 2: Global Search with Gaussians (more components)
    if time.time() < deadline - 40.0:
        for n_gauss in [5, 7]:
            if time.time() > deadline - 30.0:
                break
            n_params = 3 * n_gauss + 1
            bounds = []
            for _ in range(n_gauss):
                bounds.append((-0.25, 0.25))
                bounds.append((0.01, 0.15))
                bounds.append((0.0, 5.0))
            bounds.append((0.0, 2.0))
            
            try:
                res = differential_evolution(
                    objective_gauss, bounds, maxiter=80, popsize=20,
                    tol=1e-6, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = gaussians_to_h(res.x)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass
                
    # Phase 2.5: Global Search with Blocks
    if time.time() < deadline - 30.0:
        for n_blocks in [10, 20, 50]:
            if time.time() > deadline - 20.0:
                break
            bounds = [(0.0, 5.0)] * n_blocks
            try:
                res = differential_evolution(
                    lambda p: objective_block(p, n_blocks), bounds, maxiter=100, popsize=20,
                    tol=1e-6, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = blocks_to_h(res.x, n_blocks)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass

    # Phase 3: Local Refinement on full 600-dim space
    # Use exact objective for local refinement (it's differentiable a.e.)
    starts = [best_h]
    for _ in range(8):
        # Random smooth start
        h_rand = np.abs(rng.normal(1.0, 0.5, N))
        kernel = np.ones(10) / 10
        h_rand = np.convolve(h_rand, kernel, mode='same')
        starts.append(h_rand)
    
    for h_start in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res = opt.minimize(
                obj_exact, h_start, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 4: Final Polish with tighter tolerance
    if time.time() < deadline - 3.0:
        try:
            res = opt.minimize(
                obj_exact, best_h, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 5: Additional polish from slightly perturbed best
    if time.time() < deadline - 1.0:
        try:
            h_pert = best_h + rng.normal(0, 0.01, N)
            h_pert = np.clip(h_pert, 0, None)
            res = opt.minimize(
                obj_exact, h_pert, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    return list(best_h)
>>>>>>> REPLACE

