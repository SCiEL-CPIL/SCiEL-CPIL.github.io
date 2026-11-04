# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute all triangle areas vectorized. Returns array of shape (C(N,3),)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return None

def _score(pts):
    """Compute min triangle area / hull area."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    hull_area = _hull_area(pts)
    if hull_area is None or hull_area < 1e-9:
        return -1e9
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_value(areas, temperature):
    """Compute soft minimum using log-sum-exp trick."""
    if len(areas) == 0:
        return 0.0
    neg_areas = -areas
    max_val = np.max(neg_areas)
    if max_val > 500:
        max_val = 500
    shifted = neg_areas - max_val
    log_sum = max_val + np.log(np.sum(np.exp(shifted)))
    return -temperature * log_sum

def _soft_min_ratio(pts, temperature):
    """Compute soft min triangle area ratio."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    hull_area = _hull_area(pts)
    if hull_area is None or hull_area < 1e-9:
        return -1e9
    areas = _all_triangle_areas(pts)
    soft_min = _soft_min_value(areas, temperature)
    return soft_min / hull_area

def _neg_soft_min_ratio(pts, temperature):
    """Negative of soft min ratio for minimization."""
    return -_soft_min_ratio(pts, temperature)

def _objective_de(params):
    """Objective function for Differential Evolution."""
    pts = params.reshape(N, 2)
    # Use a fixed temperature for DE to keep it smooth but responsive
    # Temperature 0.05 is a good balance
    return _neg_soft_min_ratio(pts, 0.05)

def _refine_l_bfgs(pts, time_budget):
    """Refine points using L-BFGS-B with decreasing temperature."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Anneal temperature from 0.05 to 0.001
    temps = [0.05, 0.02, 0.01, 0.005, 0.001]
    
    for temp in temps:
        if time.time() - start_time > time_budget:
            break
            
        def objective(params):
            p = params.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp)
        
        try:
            res = minimize(
                objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'disp': False}
            )
            if res.success or res.fun < objective(best_pts.flatten()):
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    rng = np.random.default_rng(42)
    
    # Define bounds for DE: [-2, 2] is a safe range for unit-area normalized shapes
    # The scale is handled by the ratio, but DE needs finite bounds.
    # We can center the result later.
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    # Run Differential Evolution
    # popsize=20, maxiter=100 is a reasonable start. 
    # We check time frequently.
    de_result = None
    try:
        de_result = minimize(
            _objective_de,
            np.zeros(N * 2), # Start from center? Or random?
            method='differential_evolution',
            bounds=bounds,
            options={
                'maxiter': 100,
                'popsize': 20,
                'polish': False, # We do our own polish
                'init': 'latinhypercube',
                'seed': 42,
                'updating': 'immediate',
                'tol': 0.01,
                'disp': False
            }
        )
    except Exception:
        pass
    
    best_pts = None
    best_score = -1e9
    
    if de_result is not None and de_result.success:
        best_pts = de_result.x.reshape(N, 2)
        best_score = _score(best_pts)
        
        # Refine with L-BFGS
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10.0:
            best_pts, best_score = _refine_l_bfgs(best_pts, remaining - 5.0)
    else:
        # Fallback if DE fails: Use a regular polygon and refine
        best_pts = _regular_polygon(N)
        best_score = _score(best_pts)
        
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10.0:
            best_pts, best_score = _refine_l_bfgs(best_pts, remaining - 5.0)
            
    # If we have time, try a few random restarts with DE?
    # Or just stick with the best one. DE is usually robust.
    
    return best_pts

# Helper functions from original code that are still needed
def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

# EVOLVE-BLOCK-END
