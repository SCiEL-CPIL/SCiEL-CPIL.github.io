import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
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

def _soft_min_score(points, temp=0.05):
    """
    Compute a smooth approximation of the minimum triangle area.
    Uses log-sum-exp to create a differentiable surrogate for the min function.
    Returns negative score for minimization.
    """
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min: -temp * log(sum(exp(-areas/temp)))
        # We want to maximize the min area, so we minimize -soft_min_area
        # soft_min_area = -temp * log(sum(exp(-areas/temp)))
        # To avoid overflow, subtract max(areas)
        max_area = np.max(areas)
        log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / temp)))
        soft_min_area = -(max_area + temp * log_sum_exp)
        
        # We want to maximize soft_min_area / ha
        # So we minimize - (soft_min_area / ha)
        return -(soft_min_area / ha)
    except Exception:
        return 0.0

def _generate_ring_points(radii, counts, phases):
    """
    Generate points from ring parameters.
    radii: array of radii for each ring
    counts: array of number of points in each ring
    phases: array of angular offsets for each ring
    """
    pts = []
    for r, c, p in zip(radii, counts, phases):
        if c == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, c, endpoint=False) + p
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts.append(np.column_stack([x, y]))
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def _ring_score(params, counts):
    """
    Objective function for ring optimization.
    params: flattened array [r1, r2, ..., rk, p1, p2, ..., pk]
    counts: list of point counts for each ring
    """
    k = len(counts)
    radii = params[:k]
    phases = params[k:]
    
    # Ensure radii are positive
    radii = np.abs(radii)
    
    points = _generate_ring_points(radii, counts, phases)
    
    # Normalize to unit hull area
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    points = points / np.sqrt(ha)
    
    return _score(points)

def _neg_ring_score(params, counts):
    return -_ring_score(params, counts)

def _neg_soft_ring_score(params, counts, temp=0.05):
    """
    Soft version for optimization.
    """
    k = len(counts)
    radii = params[:k]
    phases = params[k:]
    
    radii = np.abs(radii)
    
    points = _generate_ring_points(radii, counts, phases)
    
    # Normalize
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    points = points / np.sqrt(ha)
    
    return _soft_min_score(points, temp)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Candidate ring structures (counts per ring)
    # Sum of counts must be 14
    structures = [
        [14],
        [7, 7],
        [8, 6],
        [9, 5],
        [10, 4],
        [11, 3],
        [12, 2],
        [13, 1],
        [4, 5, 5],
        [3, 5, 6],
        [2, 6, 6],
        [2, 5, 7],
        [3, 4, 7],
        [4, 4, 6],
        [3, 3, 8],
        [2, 4, 8],
        [2, 3, 9],
        [1, 6, 7],
        [1, 5, 8],
        [1, 4, 9],
        [1, 3, 10],
        [1, 2, 11],
        [1, 1, 12],
        [2, 2, 3, 7],
        [2, 2, 4, 6],
        [2, 2, 5, 5],
        [2, 3, 3, 6],
        [2, 3, 4, 5],
        [2, 4, 4, 4],
        [3, 3, 3, 5],
        [3, 3, 4, 4],
        [1, 2, 3, 8],
        [1, 2, 4, 7],
        [1, 2, 5, 6],
        [1, 3, 3, 7],
        [1, 3, 4, 6],
        [1, 3, 5, 5],
        [1, 4, 4, 5],
        [2, 2, 2, 8],
        [2, 2, 3, 7],
        [2, 2, 4, 6],
        [2, 2, 5, 5],
        [2, 3, 2, 7],
        [2, 3, 3, 6],
        [2, 3, 4, 5],
        [2, 4, 2, 6],
        [2, 4, 3, 5],
        [2, 4, 4, 4],
        [2, 5, 2, 5],
        [2, 5, 3, 4],
        [2, 5, 4, 3],
        [2, 6, 2, 4],
        [2, 6, 3, 3],
        [2, 7, 2, 3],
        [2, 7, 3, 2],
        [2, 8, 2, 2],
        [3, 3, 2, 6],
        [3, 3, 3, 5],
        [3, 3, 4, 4],
        [3, 4, 2, 5],
        [3, 4, 3, 4],
        [3, 4, 4, 3],
        [3, 5, 2, 4],
        [3, 5, 3, 3],
        [3, 5, 4, 2],
        [3, 6, 2, 3],
        [3, 6, 3, 2],
        [3, 7, 2, 2],
        [4, 4, 2, 4],
        [4, 4, 3, 3],
        [4, 4, 4, 2],
        [4, 5, 2, 3],
        [4, 5, 3, 2],
        [4, 6, 2, 2],
        [5, 5, 2, 2],
    ]
    
    # Remove duplicates
    structures = [tuple(sorted(s)) for s in structures]
    structures = list(set(structures))
    
    # Sort by number of rings (fewer rings is simpler, often better)
    structures.sort(key=lambda x: (len(x), x))
    
    def _try_ring_structure(counts):
        nonlocal best_score, best_points
        if time.time() > deadline:
            return
        
        k = len(counts)
        # Parameters: k radii, k phases
        # Initial guess: radii spread out, phases random
        rng = np.random.RandomState(42)
        
        # Bounds: radii [0.1, 2.0], phases [0, 2*pi]
        bounds = [(0.1, 2.0)] * k + [(0, 2 * np.pi)] * k
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            x0 = np.array([
                np.linspace(0.5, 1.5, k)[i] + rng.uniform(-0.1, 0.1) for i in range(k)
            ] + [rng.uniform(0, 2 * np.pi) for _ in range(k)])
            
            # 1. Differential Evolution for global search
            try:
                res_de = scipy.optimize.differential_evolution(
                    _neg_soft_ring_score,
                    bounds,
                    args=(counts, 0.1),
                    seed=42+trial,
                    maxiter=100,
                    tol=1e-6,
                    popsize=15,
                    polish=False
                )
                if time.time() > deadline:
                    break
                
                # 2. Local Polish with L-BFGS-B on soft score
                res_lbgfs = scipy.optimize.minimize(
                    _neg_soft_ring_score,
                    res_de.x,
                    args=(counts, 0.05),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                # 3. Final Polish with Nelder-Mead on exact score
                res_nm = scipy.optimize.minimize(
                    _neg_ring_score,
                    res_lbgfs.x,
                    args=(counts,),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                # Evaluate final points
                final_params = res_nm.x
                radii = np.abs(final_params[:k])
                phases = final_params[k:]
                points = _generate_ring_points(radii, counts, phases)
                
                ha = _hull_area(points)
                if ha < 1e-12:
                    continue
                points = points / np.sqrt(ha)
                
                sc = _score(points)
                
                if sc > best_score:
                    best_score = sc
                    best_points = points.copy()
                    
            except Exception:
                continue
    
    # Iterate through structures
    # Prioritize 2-ring and 3-ring structures as they are most likely
    # We'll just iterate through the sorted list, but maybe limit time per structure
    time_per_structure = 2.0 # seconds
    
    for counts in structures:
        if time.time() > deadline:
            break
        
        t_start = time.time()
        _try_ring_structure(counts)
        
        # If we've spent too long on this structure, move on
        # (The internal checks handle this, but this is a hard stop)
        if time.time() - t_start > time_per_structure * 2:
            continue

    # If we have time left, try to polish the best point set with raw coordinates
    # This allows points to drift off the rings if it improves the score
    if best_points is not None and time.time() < deadline - 10:
        def _neg_raw_score(points_flat):
            points = points_flat.reshape(-1, 2)
            return -_score(points)
        
        def _neg_soft_raw_score(points_flat):
            points = points_flat.reshape(-1, 2)
            return _soft_min_score(points, 0.05)
        
        try:
            # Start from the best ring-based solution
            x0 = best_points.flatten()
            
            # Soft polish
            res_soft = scipy.optimize.minimize(
                _neg_soft_raw_score,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            # Exact polish
            res_exact = scipy.optimize.minimize(
                _neg_raw_score,
                res_soft.x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            
            final_points = res_exact.x.reshape(-1, 2)
            ha = _hull_area(final_points)
            if ha > 1e-12:
                final_points = final_points / np.sqrt(ha)
                sc = _score(final_points)
                if sc > best_score:
                    best_score = sc
                    best_points = final_points.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        ha = _hull_area(best_points)
        if ha > 1e-12:
            best_points = best_points / np.sqrt(ha)

    return best_points
