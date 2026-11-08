"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _all_triangle_areas(pts):
    """Vectorized computation of all C(N,3) triangle areas."""
    n = len(pts)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    v1 = pts[j_idx] - pts[i_idx]
    v2 = pts[k_idx] - pts[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas, i_idx, j_idx, k_idx

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points (vectorized)."""
    areas, _, _, _ = _all_triangle_areas(pts)
    return np.min(areas)

def _convex_hull_area(pts):
    """Compute area of convex hull using scipy."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _is_strictly_convex(pts):
    """Check if points form a strictly convex polygon (no three collinear, all turn same direction)."""
    n = len(pts)
    if n < 3:
        return False
    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(cross > 0)
    return all(signs) or all(not s for s in signs)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _perturb_polygon(base_pts, angles_pert, radii_pert):
    """Apply angular and radial perturbations to a regular polygon."""
    n = len(base_pts)
    base_angles = np.arctan2(base_pts[:, 1], base_pts[:, 0])
    base_radii = np.sqrt(base_pts[:, 0]**2 + base_pts[:, 1]**2)
    new_angles = base_angles + angles_pert
    new_radii = base_radii + radii_pert
    new_radii = np.maximum(new_radii, 0.01)
    pts = np.column_stack([new_radii * np.cos(new_angles), new_radii * np.sin(new_angles)])
    return pts

def _local_refine(pts, max_iters=200, tol=1e-10):
    """Local refinement: iteratively move worst-performing points to improve min triangle area."""
    n = len(pts)
    pts = pts.copy()
    current_score = _score(pts)
    
    for it in range(max_iters):
        areas, i_idx, j_idx, k_idx = _all_triangle_areas(pts)
        min_area = np.min(areas)
        
        # Find the worst triple
        worst_idx = np.argmin(areas)
        i, j, k = i_idx[worst_idx], j_idx[worst_idx], k_idx[worst_idx]
        
        # Try to improve by moving each point of the worst triple
        improved = False
        for pt_idx in [i, j, k]:
            for step in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = pts.copy()
                        trial[pt_idx] += np.array([dx, dy])
                        if _is_strictly_convex(trial):
                            s = _score(trial)
                            if s > current_score + tol:
                                current_score = s
                                pts = trial
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return pts, current_score

def _gradient_polish(pts, max_iters=100, tol=1e-12):
    """Gradient-based polish using numerical gradients."""
    n = len(pts)
    pts = pts.copy()
    current_score = _score(pts)
    
    for it in range(max_iters):
        # Compute numerical gradient of score w.r.t. each point coordinate
        grad = np.zeros_like(pts)
        h = 1e-6
        for i in range(n):
            for d in range(2):
                pts_plus = pts.copy()
                pts_plus[i, d] += h
                if _is_strictly_convex(pts_plus):
                    s_plus = _score(pts_plus)
                else:
                    s_plus = current_score - 1e6
                grad[i, d] = (s_plus - current_score) / h
        
        # Gradient ascent step
        step_size = 0.001
        new_pts = pts + step_size * grad
        if _is_strictly_convex(new_pts):
            new_score = _score(new_pts)
            if new_score > current_score + tol:
                current_score = new_score
                pts = new_pts
            else:
                break
        else:
            break
    
    return pts, current_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer before 150s kill
    
    # Initialize with regular 14-gon
    best_pts = _regular_polygon(N, radius=1.0)
    best_score = _score(best_pts)
    
    def objective(params):
        """Negative score (we minimize)."""
        angles_pert = params[:N]
        radii_pert = params[N:]
        radii_pert = np.maximum(radii_pert, -0.5)
        pts = _perturb_polygon(best_pts, angles_pert, radii_pert)
        if not _is_strictly_convex(pts):
            return 1e10
        s = _score(pts)
        return -s
    
    rng = np.random.default_rng(42)
    
    # Stage 1: Coarse global search with multiple Nelder-Mead restarts
    n_restarts = 8
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
        
        if restart == 0:
            x_init = np.zeros(2 * N)
        else:
            x_init = rng.normal(0, 0.15, size=2 * N)
        
        try:
            result = minimize(
                objective,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 6000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            
            if result.fun < 1e9:
                angles_pert = result.x[:N]
                radii_pert = result.x[N:]
                radii_pert = np.maximum(radii_pert, -0.5)
                candidate_pts = _perturb_polygon(best_pts, angles_pert, radii_pert)
                
                if _is_strictly_convex(candidate_pts):
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts
        except Exception:
            continue
    
    # Stage 2: Local refinement targeting worst triples
    elapsed = time.time() - start_time
    if elapsed < time_limit - 15:
        refined_pts, refined_score = _local_refine(best_pts, max_iters=300)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # Stage 3: Gradient-based polish
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        polished_pts, polished_score = _gradient_polish(best_pts, max_iters=200)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    # Stage 4: Final scaling adjustment
    elapsed = time.time() - start_time
    if elapsed < time_limit - 1:
        centroid = np.mean(best_pts, axis=0)
        for scale in np.linspace(0.98, 0.5, 30):
            if time.time() - start_time > time_limit - 0.5:
                break
            scaled_pts = centroid + scale * (best_pts - centroid)
            if _is_strictly_convex(scaled_pts):
                s = _score(scaled_pts)
                if s > best_score:
                    best_score = s
                    best_pts = scaled_pts
    
    return best_pts
