import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools
import scipy.spatial as spat

N = 13
DEADLINE = 145.0  # Safety margin for 150s limit

# Precompute triangle indices for vectorized area calculation
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
NUM_TRIS = len(TRI_INDICES)

def _compute_triangle_areas(pts):
    """
    Vectorized computation of areas for all triples of points.
    """
    p1 = pts[TRI_INDICES[:, 0]]
    p2 = pts[TRI_INDICES[:, 1]]
    p3 = pts[TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_hard(pts):
    """
    Exact score calculation: min triangle area / convex hull area.
    """
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _compute_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_objective(params, beta):
    """
    Objective function for optimization: negative of soft-min area / hull area.
    We maximize min_area / hull_area, so we minimize -min_area / hull_area.
    """
    pts = params.reshape(N, 2)
    
    # Check for degenerate configurations
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _compute_triangle_areas(pts)
    a_min = np.min(areas)
    
    # Soft min using log-sum-exp
    shifted = -beta * (areas - a_min)
    # shifted <= 0, so exp(shifted) <= 1, no overflow
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = a_min - log_sum / beta
    
    return -soft_min / hull_area

def _generate_initial_configurations():
    """
    Generate several symmetric initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 2. 1 center + 12 ring
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts[1:] = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 3. Concentric rings: 1 + 3 + 4 + 5
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 3), (1.0, 4), (1.5, 5)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += count
    configs.append(pts)
    
    # 4. Concentric rings: 1 + 4 + 4 + 4
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 4), (1.0, 4), (1.5, 4)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += count
    configs.append(pts)
    
    # 5. Grid-like: 4x4 minus one corner, centered
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_pts.append([i - 1.5, j - 1.5])
    grid_pts = np.array(grid_pts)
    configs.append(grid_pts)
    
    # 6. Random initial points
    for _ in range(5):
        pts = np.random.randn(N, 2) * 2
        configs.append(pts)
    
    return configs

def _local_refine(pts, beta_start=10.0, beta_end=500.0, max_iter=200):
    """
    Local refinement using L-BFGS-B with increasing beta.
    """
    current = pts.flatten()
    
    betas = np.geomspace(beta_start, beta_end, 5)
    
    for beta in betas:
        try:
            res = minimize(
                lambda x: _soft_min_objective(x, beta),
                current,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current = res.x
        except Exception:
            continue
    
    return current.reshape(N, 2)

def construct():
    start_time = time.time()
    
    configs = _generate_initial_configurations()
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Local refinement from each initial configuration
    for init_pts in configs:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        try:
            refined = _local_refine(init_pts, beta_start=10.0, beta_end=500.0, max_iter=150)
            hard_score = _score_hard(refined)
            
            if hard_score > best_score:
                best_score = hard_score
                best_points = refined.copy()
        except Exception:
            continue
    
    # Phase 2: Differential Evolution for global search
    if time.time() - start_time < DEADLINE - 20:
        # Define bounds for points in a reasonable range
        bounds = [(-5, 5)] * (N * 2)
        
        def de_objective(params):
            if time.time() - start_time > DEADLINE - 10:
                return 1e10
            pts = params.reshape(N, 2)
            hull_area = _hull_area(pts)
            if hull_area < 1e-12:
                return 1e10
            areas = _compute_triangle_areas(pts)
            min_area = np.min(areas)
            return -min_area / hull_area
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                updating='immediate'
            )
            
            de_pts = de_result.x.reshape(N, 2)
            de_score = _score_hard(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish on the best point found
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        try:
            polished = _local_refine(best_points, beta_start=100.0, beta_end=2000.0, max_iter=300)
            polish_score = _score_hard(polished)
            
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = configs[0]
    
    return best_points
