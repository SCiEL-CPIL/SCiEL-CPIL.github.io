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
        # Ensure at least 3 non-collinear points
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

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp (numerically stable)
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    # Use -beta * shifted to approximate min. 
    # min(x) = -lim_{b->inf} (1/b) log(sum(exp(-b*x)))
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft-score for gradient-based optimization."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
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

    current_points = best_points.copy()
    current_score = best_score
    
    # Simulated Annealing Parameters
    # We perturb in Cartesian coordinates.
    # Initial temperature should be large enough to explore, small enough to converge.
    # Score is roughly 0.048. Area differences are small.
    T = 0.005 
    T_min = 1e-6
    cooling_rate = 0.9995
    
    rng = np.random.RandomState(42)
    
    # SA Loop
    # Each evaluation takes ~1-2ms. 
    # 140s / 0.002s ~ 70,000 iterations.
    # We will run until time is up.
    
    last_polish_time = start_time
    polish_interval = 5.0 # seconds
    
    while _time_left() > 5.0:
        # Generate perturbation
        # Use Gaussian noise scaled by T
        # The magnitude of perturbation should be related to the scale of the problem.
        # Points are in a unit area hull, so coords are ~1.
        # A step size of ~0.1 * T_factor is reasonable.
        step_size = T * 10.0 
        if step_size < 1e-4:
            step_size = 1e-4
            
        noise = rng.normal(0, step_size, current_points.shape)
        new_points = current_points + noise
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            _update_best(new_points, new_score)
        else:
            # Metropolis criterion
            if T > T_min:
                delta = current_score - new_score
                if delta < 1e-12:
                    prob = 1.0
                else:
                    prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Periodic Polish with Nelder-Mead
        # This helps escape the "noise" of SA and find the precise local optimum
        if time.time() - last_polish_time > polish_interval:
            last_polish_time = time.time()
            if _time_left() > 10.0:
                try:
                    # Use Nelder-Mead on the exact score
                    # It is robust to the non-differentiability of min()
                    res = scipy.optimize.minimize(
                        _neg_score,
                        current_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    polished_points = res.x.reshape(-1, 2)
                    polished_score = _score(polished_points)
                    _update_best(polished_points, polished_score)
                    
                    # Update current state to the polished result if it's better
                    if polished_score > current_score:
                        current_points = polished_points
                        current_score = polished_score
                        
                    # Reset temperature slightly to explore around the new basin
                    # But don't reset it fully, maintain progress
                    T = max(T, 0.001) 
                except Exception:
                    pass

    # Final Polish with high precision
    if _time_left() > 2.0:
        try:
            # Use soft surrogate with high beta for final smooth gradient step
            # This can refine the solution further if NM stopped early due to non-smoothness
            res_final = scipy.optimize.minimize(
                lambda x: _neg_soft_score(x, beta=500.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
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
