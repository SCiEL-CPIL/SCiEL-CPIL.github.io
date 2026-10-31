import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools
import scipy.spatial as spat
import sys

N = 13
DEADLINE = 148.0  # Safety margin for 150s limit

# Precompute triangle indices
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
NUM_TRIS = len(TRI_INDICES)

def _triangle_areas(pts):
    """Vectorized triangle areas for all triples."""
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

def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    return np.min(areas) / hull_area

def _soft_min_objective(params, beta):
    """
    Smooth surrogate: maximize soft-min area / hull area.
    Returns negative for minimization.
    """
    pts = params.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    
    # Soft min via log-sum-exp
    shifted = -beta * (areas - a_min)
    # Clip to avoid underflow issues
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = a_min - log_sum / beta
    
    return -soft_min / hull_area

def _hard_objective(params):
    """Hard objective for DE: negative min_area / hull_area."""
    pts = params.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas(pts)
    return -np.min(areas) / hull_area

def _make_regular_polygon(n, radius=1.0, center=None):
    """Create regular n-gon."""
    if center is None:
        center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return pts + center

def _make_ring_config(inner_count, outer_count, inner_radius=0.5, outer_radius=1.0):
    """Create concentric ring configuration."""
    total = inner_count + outer_count
    if total != N:
        return None
    
    pts = np.zeros((N, 2))
    idx = 0
    
    if inner_count > 0:
        if inner_count == 1:
            pts[idx] = [0, 0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, inner_count, endpoint=False)
            pts[idx:idx+inner_count] = inner_radius * np.column_stack([np.cos(angles), np.sin(angles)])
            idx += inner_count
    
    if outer_count > 0:
        angles = np.linspace(0, 2 * np.pi, outer_count, endpoint=False)
        pts[idx:idx+outer_count] = outer_radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    return pts

def _local_refine(pts, beta_schedule=None, max_iter=300):
    """
    Multi-stage local refinement with increasing beta.
    """
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    current = pts.flatten()
    
    for beta in beta_schedule:
        try:
            res = minimize(
                lambda x: _soft_min_objective(x, beta),
                current,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < 1e9:
                current = res.x
        except Exception:
            continue
    
    return current.reshape(N, 2)

def _generate_starts():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 13-gon
    configs.append(_make_regular_polygon(13))
    
    # 2. 1 center + 12 ring
    configs.append(_make_ring_config(1, 12))
    
    # 3. 1 + 6 + 6 (hexagonal)
    cfg = _make_ring_config(1, 6, inner_radius=0.5, outer_radius=1.0)
    if cfg is not None:
        # Add 6 more on outer ring
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer2 = 1.0 * np.column_stack([np.cos(angles), np.sin(angles)])
        # Need to adjust - let's do 1 + 6 + 6 properly
        pts = np.zeros((N, 2))
        pts[0] = [0, 0]
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        pts[1:7] = 0.6 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        pts[7:13] = 1.0 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        configs.append(pts)
    
    # 4. 1 + 4 + 4 + 4 rings
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    for r, count, offset in [(0.4, 4, 0), (0.7, 4, np.pi/4), (1.0, 4, 0)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + offset
        idx_start = len(pts) - count  # This won't work, let me redo
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count, offset in [(0.4, 4, 0), (0.7, 4, np.pi/4), (1.0, 4, 0)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + offset
        pts[idx:idx+count] = r * np.column_stack([np.cos(angles), np.sin(angles)])
        idx += count
    configs.append(pts)
    
    # 5. Grid 4x4 minus corner
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            grid_pts.append([(i - 1.5) * 0.7, (j - 1.5) * 0.7])
    configs.append(np.array(grid_pts))
    
    # 6. Grid 4x4 minus center
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i in [1,2] and j in [1,2] and (i,j) in [(1,1),(1,2),(2,1),(2,2)]:
                continue
            grid_pts.append([(i - 1.5) * 0.7, (j - 1.5) * 0.7])
    if len(grid_pts) == 13:
        configs.append(np.array(grid_pts))
    
    # 7. 3x5 grid minus 2
    grid_pts = []
    for i in range(5):
        for j in range(3):
            if (i, j) in [(0,0), (4,2)]:
                continue
            grid_pts.append([(i - 2) * 0.6, (j - 1) * 0.6])
    if len(grid_pts) == 13:
        configs.append(np.array(grid_pts))
    
    # 8. 5x3 grid minus 2 (transposed)
    grid_pts = []
    for i in range(3):
        for j in range(5):
            if (i, j) in [(0,0), (2,4)]:
                continue
            grid_pts.append([(i - 1) * 0.6, (j - 2) * 0.6])
    if len(grid_pts) == 13:
        configs.append(np.array(grid_pts))
    
    # 9. Perturbed regular 13-gon
    base = _make_regular_polygon(13)
    np.random.seed(42)
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        perturbed = base + np.random.randn(N, 2) * 0.1
        configs.append(perturbed)
    
    # 10. Random configurations
    np.random.seed(123)
    for seed in [10, 20, 30, 40, 50]:
        np.random.seed(seed)
        pts = np.random.randn(N, 2) * 1.5
        configs.append(pts)
    
    # Score and keep top
    scored = []
    for cfg in configs:
        try:
            s = _score(cfg)
            scored.append((s, cfg))
        except:
            pass
    
    scored.sort(key=lambda x: -x[0])
    return [cfg for _, cfg in scored[:10]]

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    configs = _generate_starts()
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Local refinement from each start
    for init_pts in configs:
        if remaining() < 50:
            break
        try:
            refined = _local_refine(init_pts)
            hard_score = _score(refined)
            if hard_score > best_score:
                best_score = hard_score
                best_points = refined.copy()
        except Exception:
            continue
    
    # Phase 2: DE global search
    if remaining() > 35 and best_points is not None:
        bounds = [(-4, 4)] * (N * 2)
        
        def de_obj(params):
            if remaining() < 15:
                return 1e10
            return _hard_objective(params)
        
        try:
            de_result = differential_evolution(
                de_obj,
                bounds,
                maxiter=60,
                popsize=12,
                tol=1e-6,
                seed=42,
                polish=False,
                updating='immediate'
            )
            
            de_pts = de_result.x.reshape(N, 2)
            de_score = _score(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts.copy()
            
            # Refine DE result
            if remaining() > 15:
                refined_de = _local_refine(de_pts)
                refined_score = _score(refined_de)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_de.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish
    if best_points is not None and remaining() > 5:
        try:
            polished = _local_refine(best_points, beta_schedule=[100, 200, 400, 800], max_iter=500)
            polish_score = _score(polished)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = configs[0]
    
    return best_points
