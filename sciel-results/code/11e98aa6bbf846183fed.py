<<<<<<< SEARCH
def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=30),
                x0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if _score(res.x.reshape(N, 2)) > best_s:
                best_s = _score(res.x.reshape(N, 2))
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Hard objective polish
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.copy()
        except Exception:
            pass
    
    return best_x, best_s
=======
def _coordinate_descent_polish(points, deadline, max_iter=500):
    """Direct coordinate perturbation local search."""
    best = points.copy()
    best_s = _score(best)
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Try perturbing one random point
        idx = rng.integers(0, N)
        step = 0.01 * (1.0 - 0.5 * it / max_iter)
        
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            new_pts = best.copy()
            delta = rng.normal(0, step, size=2)
            new_pts[idx] += delta
            s = _score(new_pts)
            if s > best_s:
                best = new_pts.copy()
                best_s = s
    
    return best, best_s

def _simulated_annealing_polish(points, deadline, t_start=1.0, t_end=0.01, max_iter=200):
    """Simulated annealing to escape local minima."""
    best = points.copy()
    best_s = _score(best)
    current = points.copy()
    current_s = best_s
    rng = np.random.default_rng(7)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Perturb a random point
        idx = rng.integers(0, N)
        step = 0.05 * t + 0.005
        new_pts = current.copy()
        new_pts[idx] += rng.normal(0, step, size=2)
        
        s_new = _score(new_pts)
        
        # Accept if better or with probability exp((s_new - current_s) / t)
        if s_new > current_s:
            current = new_pts
            current_s = s_new
            if s_new > best_s:
                best = new_pts.copy()
                best_s = s_new
        else:
            prob = np.exp((s_new - current_s) / (t + 1e-10))
            if rng.random() < prob:
                current = new_pts
                current_s = s_new
    
    return best, best_s

def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=30),
                x0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if _score(res.x.reshape(N, 2)) > best_s:
                best_s = _score(res.x.reshape(N, 2))
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Hard objective polish
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 8:
        cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
        if cd_s > best_s:
            best_s = cd_s
            best_x = cd_pts.flatten()
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5:
        sa_pts, sa_s = _simulated_annealing_polish(best_x.reshape(N, 2), deadline)
        if sa_s > best_s:
            best_s = sa_s
            best_x = sa_pts.flatten()
    
    return best_x, best_s
>>>>>>> REPLACE

