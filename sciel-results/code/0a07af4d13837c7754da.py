# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns (areas, indices)."""
    n = len(pts)
    idx = []
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                idx.append((i, j, k))
                areas.append(a)
    return np.array(areas), idx


def _score(pts):
    """Compute the Heilbronn score: A_min / A_hull."""
    n = len(pts)
    if n < 3:
        return 0.0
    areas, idx = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    if a_min < 1e-15:
        return 0.0
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull < 1e-15:
        return 0.0
    return a_min / a_hull


def _delaunay_forces(pts):
    """
    Compute forces based on Delaunay triangulation.
    For each Delaunay triangle, if its area is small, push vertices apart.
    Also add repulsive forces between all point pairs to avoid clustering.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Delaunay-based forces
    try:
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            i, j, k = simplex
            area = _triangle_area(pts[i], pts[j], pts[k])
            if area < 1e-15:
                continue
            
            # Centroid of the triangle
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            
            # Push vertices away from centroid, stronger for smaller triangles
            strength = 1.0 / (area + 0.01)
            
            for v in [i, j, k]:
                diff = pts[v] - centroid
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    forces[v] += strength * diff / norm * 0.05
    except Exception:
        pass
    
    # Pairwise repulsive forces (Coulomb-like)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                forces[i] += np.random.randn(2) * 0.01
                forces[j] -= np.random.randn(2) * 0.01
            else:
                repulsion = 0.1 / (dist * dist + 0.01)
                forces[i] += repulsion * diff / dist
                forces[j] -= repulsion * diff / dist
    
    return forces


def _laplacian_smoothing(pts, weights=None):
    """
    Apply Laplacian smoothing to move points toward the average of their neighbors.
    """
    n = len(pts)
    if weights is None:
        weights = np.ones((n, n))
    
    smoothed = np.zeros_like(pts)
    for i in range(n):
        total_weight = 0
        weighted_sum = np.zeros(2)
        for j in range(n):
            if i != j:
                w = weights[i, j]
                weighted_sum += w * pts[j]
                total_weight += w
        if total_weight > 1e-10:
            smoothed[i] = weighted_sum / total_weight
        else:
            smoothed[i] = pts[i]
    
    return smoothed


def _compute_weights(pts):
    """
    Compute distance-based weights for Laplacian smoothing.
    """
    n = len(pts)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(pts[i] - pts[j])
                weights[i, j] = 1.0 / (dist + 0.1)
    return weights


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area, using a hybrid approach combining physics-based
    relaxation with gradient-based optimization (soft-min surrogate).
    """
    deadline = time.time() + 580.0
    n = 13
    
    # Precompute triangle indices for fast evaluation
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    
    def get_areas(points):
        """Compute areas of all 286 triangles efficiently."""
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def get_hull_area(points):
        """Compute area of the convex hull."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0
    
    def soft_min(areas, beta):
        """
        Stable soft-min approximation using log-sum-exp.
        """
        if len(areas) == 0:
            return 0.0
        m = np.min(areas)
        if m < 1e-12:
            return 0.0
        shifted = areas - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return m
        return m - (1.0 / beta) * np.log(sum_exp)
    
    def objective(params, beta):
        """
        Objective function to minimize: - (soft_min_area / hull_area)
        """
        points = params.reshape((n, 2))
        
        if not np.all(np.isfinite(points)):
            return 1e10
            
        hull_area = get_hull_area(points)
        if hull_area < 1e-9:
            return 1e10
            
        areas = get_areas(points)
        min_area_soft = soft_min(areas, beta)
        
        if min_area_soft < 1e-12:
            return 1e10
            
        ratio = min_area_soft / hull_area
        return -ratio
    
    def neg_hard_objective(params):
        """Hard objective for final polish."""
        points = params.reshape((n, 2))
        if not np.all(np.isfinite(points)):
            return 1e10
        hull_area = get_hull_area(points)
        if hull_area < 1e-9:
            return 1e10
        areas = get_areas(points)
        min_area = np.min(areas)
        if min_area < 1e-12:
            return 1e10
        return -(min_area / hull_area)
    
    # Initial configurations
    rng = np.random.default_rng(seed=42)
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init1 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Random points in a square
    init2 = rng.uniform(-1, 1, (n, 2))
    
    # 3. Random points in a disk
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    init3 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # 4. Jittered regular polygon
    init4 = init1 + rng.normal(0, 0.1, (n, 2))
    
    # 5. Grid-like pattern
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < 13:
                grid_pts.append([i * 0.8 - 1.2, j * 0.8 - 1.2])
    init5 = np.array(grid_pts[:13])
    
    # 6. Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = []
    for i in range(n):
        r = np.sqrt(i + 1) * 0.3
        theta = i * golden_angle
        fib_pts.append([r * np.cos(theta), r * np.sin(theta)])
    init6 = np.array(fib_pts)
    
    initializations = [init1, init2, init3, init4, init5, init6]
    
    best_global_params = None
    best_global_score = -1e10
    
    # Phase 1: Physics-based relaxation for each initial config
    for init_pts in initializations:
        if time.time() > deadline - 120:
            break
        
        pts = init_pts.copy()
        current_score = _score(pts)
        
        # Physics relaxation loop
        step_size = 0.03
        cooling = 1.0
        smooth_alpha = 0.1
        
        max_iters = 3000
        for it in range(max_iters):
            if time.time() > deadline - 90:
                break
            
            forces = _delaunay_forces(pts)
            weights = _compute_weights(pts)
            smoothed = _laplacian_smoothing(pts, weights)
            
            pts_new = pts + step_size * cooling * forces + smooth_alpha * (smoothed - pts)
            
            max_norm = np.max(np.linalg.norm(pts_new, axis=1))
            if max_norm > 5:
                pts_new = pts_new / max_norm * 5
            
            if np.any(np.isfinite(pts_new) == False):
                pts_new = pts.copy()
            
            new_score = _score(pts_new)
            
            if new_score > current_score:
                current_score = new_score
                pts = pts_new
            else:
                cooling *= 0.97
                step_size *= 0.995
                smooth_alpha *= 0.99
            
            if cooling < 0.001 or step_size < 0.001:
                break
        
        # Phase 2: Gradient-based optimization with soft-min surrogate
        if time.time() > deadline - 60:
            break
        
        beta_schedule = [10, 50, 100, 200, 500, 1000, 2000, 5000]
        current_params = pts.flatten()
        
        for beta in beta_schedule:
            if time.time() > deadline - 30:
                break
            
            try:
                res = minimize(
                    objective,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                cand_points = res.x.reshape((n, 2))
                hull_area = get_hull_area(cand_points)
                if hull_area < 1e-9:
                    continue
                
                areas = get_areas(cand_points)
                min_area = np.min(areas)
                score = min_area / hull_area
                
                if score > best_global_score:
                    best_global_score = score
                    best_global_params = res.x.copy()
                
                current_params = res.x.copy()
                
            except Exception:
                pass
        
        # Phase 3: Final polish with hard objective
        if time.time() > deadline - 15:
            break
        
        if best_global_params is not None:
            try:
                res_final = minimize(
                    neg_hard_objective,
                    best_global_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                final_points = res_final.x.reshape((n, 2))
                hull_area = get_hull_area(final_points)
                if hull_area > 1e-9:
                    areas = get_areas(final_points)
                    min_area = np.min(areas)
                    final_score = min_area / hull_area
                    
                    if final_score > best_global_score:
                        best_global_score = final_score
                        best_global_params = res_final.x.copy()
                        
            except Exception:
                pass
    
    # Ensure valid output
    if best_global_params is None:
        best_global_params = init1.flatten()
    
    final_points = best_global_params.reshape((n, 2))
    
    if not np.all(np.isfinite(final_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    try:
        hull = ConvexHull(final_points)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return final_points


# EVOLVE-BLOCK-END
