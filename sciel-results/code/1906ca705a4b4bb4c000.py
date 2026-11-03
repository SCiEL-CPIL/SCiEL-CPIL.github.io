"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    return np.concatenate(parts)

def _min_triangle_area(pts):
    """Compute the minimum triangle area."""
    areas = _all_triangle_areas(pts)
    return np.min(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a

def _soft_min_triangle_area(pts, alpha=30.0):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    min_a = np.min(areas)
    if min_a < 1e-15:
        return 0.0
    
    shifted = -alpha * (areas - min_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted))) - alpha * min_a
    return -log_sum / alpha

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _neg_soft_score(params, alpha=30.0):
    pts = params.reshape(-1, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    return -_soft_min_triangle_area(pts, alpha) / hull_a

def _generate_initializations(n_init=20, seed=42):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(seed)
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(regular.copy())
    
    # 2. Center + 12-gon
    for r_center in [0.0, 0.2, 0.3]:
        pts = np.zeros((N, 2))
        pts[0] = [r_center, 0]
        angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
        pts[1:] = np.column_stack([np.cos(angles12), np.sin(angles12)])
        inits.append(pts.copy())
    
    # 3. Two concentric rings: 5 inner, 8 outer
    for r1 in [0.3, 0.4, 0.5]:
        pts = np.zeros((N, 2))
        angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        pts[:5] = np.column_stack([r1*np.cos(angles5), r1*np.sin(angles5)])
        angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        pts[5:] = np.column_stack([np.cos(angles8), np.sin(angles8)])
        inits.append(pts.copy())
    
    # 4. Two rings: 7 inner, 6 outer
    for r1 in [0.4, 0.5]:
        pts = np.zeros((N, 2))
        angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        pts[:7] = np.column_stack([r1*np.cos(angles7), r1*np.sin(angles7)])
        angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        pts[7:] = np.column_stack([np.cos(angles6), np.sin(angles6)])
        inits.append(pts.copy())
    
    # 5. Three rings: 3, 4, 6
    for r1, r2 in [(0.2, 0.5), (0.3, 0.6)]:
        pts = np.zeros((N, 2))
        angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
        pts[:3] = np.column_stack([r1*np.cos(angles3), r1*np.sin(angles3)])
        angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        pts[3:7] = np.column_stack([r2*np.cos(angles4), r2*np.sin(angles4)])
        angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        pts[7:] = np.column_stack([np.cos(angles6), np.sin(angles6)])
        inits.append(pts.copy())
    
    # 6. Random points in unit disk
    for _ in range(8):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.1, 1.0, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        inits.append(pts)
    
    # 7. Slightly perturbed regular polygon
    for _ in range(3):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.uniform(-0.1, 0.1, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        inits.append(pts)
    
    return inits

def _targeted_worst_triangle_opt(pts, max_iter=50, step_size=0.05):
    """
    Iteratively find the worst (minimum area) triangle and perturb its vertices
    to increase the area, while trying not to decrease other areas too much.
    """
    pts = pts.copy()
    n = len(pts)
    
    for iteration in range(max_iter):
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        if min_area < 1e-10:
            break
        
        # Find which triangle this is
        # We need to reconstruct the triangle indices from the flat index
        # This is complex, so let's use a different approach:
        # Find all triangles and identify the minimum one
        
        # Actually, let's just do a simple local search:
        # For each point, try small perturbations and keep if score improves
        best_score = _score(pts)
        improved = False
        
        for i in range(n):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[i, 0] += dx
                    new_pts[i, 1] += dy
                    new_score = _score(new_pts)
                    if new_score > best_score + 1e-10:
                        pts = new_pts
                        best_score = new_score
                        improved = True
        
        if not improved:
            break
    
    return pts

def _refine_with_soft_min(pts, alpha=30.0, maxiter=500):
    """Refine using smooth objective."""
    try:
        res = minimize(
            lambda p: _neg_soft_score(p, alpha),
            pts.flatten(),
            method='Powell',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return pts

def _refine_with_nelder_mead(pts, maxiter=1000):
    """Refine using Nelder-Mead on exact objective."""
    try:
        res = minimize(
            _neg_score,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Generate initializations
    inits = _generate_initializations(n_init=25, seed=42)
    
    # Phase 1: Evaluate all inits and refine the top few
    init_scores = []
    for i, pts in enumerate(inits):
        s = _score(pts)
        init_scores.append((s, i, pts))
    
    init_scores.sort(key=lambda x: -x[0])
    
    # Refine top 8 initializations
    for rank, (s, idx, pts) in enumerate(init_scores[:8]):
        if time.time() > deadline - 30.0:
            break
        
        # Step 1: Soft min refinement
        refined_soft = _refine_with_soft_min(pts, alpha=30.0, maxiter=300)
        score_soft = _score(refined_soft)
        
        if score_soft > best_score:
            best_score = score_soft
            best_pts = refined_soft.copy()
        
        # Step 2: Targeted worst triangle optimization
        if time.time() < deadline - 20.0:
            targeted = _targeted_worst_triangle_opt(refined_soft, max_iter=30, step_size=0.03)
            score_targeted = _score(targeted)
            
            if score_targeted > best_score:
                best_score = score_targeted
                best_pts = targeted.copy()
        
        # Step 3: Nelder-Mead refinement
        if time.time() < deadline - 15.0:
            nm = _refine_with_nelder_mead(best_pts, maxiter=500)
            score_nm = _score(nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_pts = nm.copy()
    
    # Phase 2: More targeted optimization from best result
    if time.time() < deadline - 20.0 and best_score > 0:
        # Multiple rounds of targeted optimization with decreasing step size
        for step in [0.05, 0.03, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 10.0:
                break
            targeted = _targeted_worst_triangle_opt(best_pts, max_iter=40, step_size=step)
            score_t = _score(targeted)
            if score_t > best_score:
                best_score = score_t
                best_pts = targeted.copy()
        
        # Final Nelder-Mead polish
        if time.time() < deadline - 5.0:
            nm = _refine_with_nelder_mead(best_pts, maxiter=800)
            score_nm = _score(nm)
            if score_nm > best_score:
                best_score = score_nm
                best_pts = nm.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
