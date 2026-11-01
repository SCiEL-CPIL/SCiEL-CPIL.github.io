import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _compute_score(params):
    """
    Compute score from ring parameters.
    params: [r1, r2, ..., rm, theta1_1, theta1_2, ..., theta1_n1, theta2_1, ...]
    """
    try:
        n_rings = len(params) // 2  # first half are radii
        # Actually, let's use a fixed structure: 
        # We'll try 2-ring and 3-ring configurations
        # For now, use a flexible approach
        pass
    except Exception:
        return 0.0
    return 0.0

def _neg_score_ring(params):
    """Negative score for ring parameterization."""
    return -_compute_score(params)

def _generate_two_ring_points(r_outer, r_inner, n_outer, n_inner, phase_outer=0.0, phase_inner=0.0):
    """
    Generate points on two concentric rings.
    Outer ring: n_outer points at radius r_outer
    Inner ring: n_inner points at radius r_inner
    """
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + phase_outer
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + phase_inner
    
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([pts_outer, pts_inner])

def _score_two_ring(params):
    """
    Score for two-ring configuration.
    params: [r_outer, r_inner, phase_outer, phase_inner]
    n_outer and n_inner are fixed based on N.
    """
    try:
        r_outer, r_inner, phase_outer, phase_inner = params
        if r_outer < 1e-6 or r_inner < 1e-6:
            return 0.0
        
        # Split N points between two rings
        # Try different splits
        # For N=14, good splits might be 7+7, 8+6, 9+5, etc.
        # We'll use 7+7 as default, but optimize over splits
        n_outer = N // 2
        n_inner = N - n_outer
        
        points = _generate_two_ring_points(r_outer, r_inner, n_outer, n_inner, phase_outer, phase_inner)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_two_ring(params):
    """Negative score for two-ring optimization."""
    return -_score_two_ring(params)

def _generate_flexible_ring_points(params):
    """
    Generate points from flexible ring parameters.
    params: [r1, n1, r2, n2, ..., r_k, n_k, phase1, phase2, ..., phase_k]
    """
    try:
        # Parse parameters
        # First, determine number of rings
        # We'll use a fixed number of rings for simplicity
        n_rings = 2
        
        # Parse radii and counts
        radii = []
        counts = []
        idx = 0
        for i in range(n_rings):
            r = params[idx]
            n = int(params[idx + 1])
            radii.append(r)
            counts.append(n)
            idx += 2
        
        # Parse phases
        phases = []
        for i in range(n_rings):
            phases.append(params[idx])
            idx += 1
        
        # Verify total count
        total = sum(counts)
        if total != N:
            return None
        
        # Generate points
        all_points = []
        for i in range(n_rings):
            angles = np.linspace(0, 2 * np.pi, counts[i], endpoint=False) + phases[i]
            pts = np.column_stack([radii[i] * np.cos(angles), radii[i] * np.sin(angles)])
            all_points.append(pts)
        
        return np.vstack(all_points)
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # Initial guess: Regular 14-gon
    init_points = _regular_polygon(N, 1.0)
    init_score = _score(init_points)
    _update_best(init_points, init_score)
    
    if best_points is None:
        return _regular_polygon(N, 1.0)

    rng = np.random.RandomState(42)
    
    # Strategy 1: Optimize two-ring configurations with differential evolution
    # Try different splits of points between rings
    splits_to_try = [(7, 7), (8, 6), (9, 5), (10, 4)]
    
    for n_outer, n_inner in splits_to_try:
        if _time_left() < 15.0:
            break
        
        def _score_ring_for_split(params):
            try:
                r_outer, r_inner, phase_outer, phase_inner = params
                if r_outer < 1e-6 or r_inner < 1e-6:
                    return 0.0
                points = _generate_two_ring_points(r_outer, r_inner, n_outer, n_inner, phase_outer, phase_inner)
                return _score(points)
            except Exception:
                return 0.0
        
        def _neg_score_ring_for_split(params):
            return -_score_ring_for_split(params)
        
        # Differential evolution bounds
        bounds = [(0.1, 2.0), (0.0, 2.0), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
        
        # Initial population
        x0 = np.array([1.0, 0.5, 0.0, 0.0])
        
        try:
            if _time_left() > 20.0:
                res = scipy.optimize.differential_evolution(
                    _neg_score_ring_for_split,
                    bounds,
                    seed=rng.randint(2**31),
                    maxiter=100,
                    tol=1e-8,
                    polish=True,
                    init='sobol'
                )
                if res.success or res.fun < 0:
                    optimal_params = res.x
                    points = _generate_two_ring_points(
                        optimal_params[0], optimal_params[1],
                        n_outer, n_inner,
                        optimal_params[2], optimal_params[3]
                    )
                    sc = _score(points)
                    _update_best(points, sc)
        except Exception:
            pass
    
    # Strategy 2: Simulated annealing on full 28D space starting from best ring solution
    if _time_left() > 10.0:
        current_points = best_points.copy()
        current_score = best_score
        
        T = 0.003
        T_min = 1e-7
        cooling_rate = 0.9998
        
        last_polish_time = time.time()
        polish_interval = 3.0
        
        while _time_left() > 5.0:
            step_size = T * 8.0
            if step_size < 1e-5:
                step_size = 1e-5
            
            noise = rng.normal(0, step_size, current_points.shape)
            new_points = current_points + noise
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                _update_best(new_points, new_score)
            else:
                if T > T_min:
                    delta = current_score - new_score
                    if delta < 1e-12:
                        prob = 1.0
                    else:
                        prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
            
            T *= cooling_rate
            
            if time.time() - last_polish_time > polish_interval:
                last_polish_time = time.time()
                if _time_left() > 8.0:
                    try:
                        res = scipy.optimize.minimize(
                            lambda x: -_score(x.reshape(-1, 2)),
                            current_points.flatten(),
                            method='Nelder-Mead',
                            options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11}
                        )
                        polished_points = res.x.reshape(-1, 2)
                        polished_score = _score(polished_points)
                        _update_best(polished_points, polished_score)
                        
                        if polished_score > current_score:
                            current_points = polished_points
                            current_score = polished_score
                    except Exception:
                        pass
    
    # Final polish
    if _time_left() > 3.0 and best_points is not None:
        try:
            res_final = scipy.optimize.minimize(
                lambda x: -_score(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            final_points = res_final.x.reshape(-1, 2)
            final_score = _score(final_points)
            _update_best(final_points, final_score)
        except Exception:
            pass

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    else:
        best_points = _normalize(best_points)
        
    return best_points
