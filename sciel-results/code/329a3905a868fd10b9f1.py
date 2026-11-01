import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
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

def _triangle_areas(points):
    """Compute all triangle areas (unnormalized, 0.5*|cross|)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    return _triangle_areas(points).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calc_ratio(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min(values, tau):
    """
    Soft-min using log-sum-exp trick.
    As tau -> 0, approaches true min.
    """
    values = np.asarray(values)
    # For stability, subtract max before exp
    v_max = values.max()
    if v_max < 1e-15:
        return 0.0
    # soft_min = -tau * log(sum(exp(-values/tau)))
    # = -tau * log(sum(exp(-(values - v_max)/tau))) - v_max
    # = -tau * log(sum(exp(-(values - v_max)/tau))) - v_max
    scaled = -(values - v_max) / tau
    # Clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return v_max - tau * log_sum

def _neg_smooth_score(points_flat, tau):
    """
    Smooth surrogate objective: minimize -soft_min(triangle_areas) / hull_area.
    This provides gradient flow for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    try:
        areas = _triangle_areas(points)
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 1.0
        soft_min_val = _soft_min(areas, tau)
        score = soft_min_val / hull_area
        return -score
    except Exception:
        return 1.0

def _neg_score(points_flat):
    """Exact objective for final evaluation."""
    points = points_flat.reshape(-1, 2)
    score = calc_ratio(points)
    return -score

def normalize_points(points):
    """
    Scale points so that the convex hull area is 1.0.
    """
    try:
        area = _hull_area(points)
        if area < 1e-10:
            return points
        scale_factor = 1.0 / np.sqrt(area)
        return points * scale_factor
    except Exception:
        return points

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # 1. Global Search using Differential Evolution
    # This helps escape local minima that trap gradient-based methods.
    # We optimize the shape. The scale is invariant, so we just need to find a good shape.
    # Bounds: [-2, 2] is generous.
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    def de_objective(x):
        points = x.reshape(-1, 2)
        # Normalize to prevent drift to boundaries? 
        # Actually, calc_ratio is scale invariant. 
        # But if points are very small, numerical issues. If very large, also.
        # Normalizing inside the objective makes the landscape "flat" in radial direction.
        # This is okay for DE, but might slow convergence.
        # Let's just compute the ratio directly.
        return -calc_ratio(points)
        
    try:
        # popsize=10 -> 280 individuals. maxiter=150 -> 42000 evals.
        # Might be too slow if each eval is >1ms.
        # Let's start with smaller popsize and iterations.
        res_de = scipy.optimize.differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=10, 
            tol=1e-6,
            polish=False # We will polish manually
        )
        
        if res_de.success or res_de.fun < 0: # fun is negative score
            de_points = res_de.x.reshape(-1, 2)
            de_points = normalize_points(de_points)
            score = calc_ratio(de_points)
            if score > best_score:
                best_score = score
                best_points = de_points.copy()
    except Exception:
        pass

    # 2. Local Polish using L-BFGS-B with Soft-Min
    # Start from the best point found so far, or a regular polygon if DE failed.
    if best_points is None:
        best_points = normalize_points(_regular_polygon(N, 1.0))
    
    current_points = best_points.copy()
    
    # Anneal tau for better convergence
    tau_schedule = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    for tau in tau_schedule:
        if time.time() > deadline - 10:
            break
            
        try:
            res_lbfgs = scipy.optimize.minimize(
                lambda x: _neg_smooth_score(x, tau),
                current_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            optimized_points = res_lbfgs.x.reshape(-1, 2)
            optimized_points = normalize_points(optimized_points)
            
            score = calc_ratio(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                current_points = optimized_points
            else:
                # If no improvement, keep current_points for next tau? 
                # Or try to restart from best_points?
                # Usually, if L-BFGS fails to improve, it's stuck.
                # Let's stick with the best found so far.
                pass
                
        except Exception:
            pass

    # 3. Final Polish with Nelder-Mead (Robust to non-smoothness)
    if time.time() < deadline - 5:
        try:
            res_nm = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            nm_points = res_nm.x.reshape(-1, 2)
            nm_points = normalize_points(nm_points)
            score = calc_ratio(nm_points)
            if score > best_score:
                best_score = score
                best_points = nm_points
        except Exception:
            pass

    # Final normalization and return
    if best_points is None:
        best_points = normalize_points(_regular_polygon(N, 1.0))
    else:
        best_points = normalize_points(best_points)
        
    return best_points
