import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# Precompute indices for all triples
triples = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
idx_i, idx_j, idx_k = triples[:, 0], triples[:, 1], triples[:, 2]
n_triples = len(triples)

def _hull_area(points):
    """Compute convex hull area using scipy's ConvexHull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Compute all triangle areas vectorized."""
    x = points[:, 0]
    y = points[:, 1]
    v1x = x[idx_j] - x[idx_i]
    v1y = y[idx_j] - y[idx_i]
    v2x = x[idx_k] - x[idx_i]
    v2y = y[idx_k] - y[idx_i]
    cross = v1x * v2y - v1y * v2x
    return 0.5 * np.abs(cross)

def compute_score(points):
    """
    Compute the score: min_triangle_area / convex_hull_area
    for given points (N, 2).
    """
    area_hull = _hull_area(points)
    if area_hull < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / area_hull

def _soft_min_objective(points_flat, beta=50.0):
    """
    Smooth surrogate objective for L-BFGS-B.
    Uses log-sum-exp to approximate the minimum triangle area.
    Returns negative soft score for minimization.
    """
    points = points_flat.reshape(-1, 2)
    area_hull = _hull_area(points)
    if area_hull < 1e-9:
        return 1e6
    
    areas = _triangle_areas(points)
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * a_i))) / beta approximates min(a_i)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    score = soft_min_area / area_hull
    return -score

def _hard_objective(points_flat):
    """Negative hard score for Nelder-Mead refinement."""
    points = points_flat.reshape(-1, 2)
    return -compute_score(points)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate points of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We should stop searching by 145s to be safe.
    time_limit = 145.0
    
    # Initial guesses: diverse configurations including concentric rings
    inits = []
    
    # 1. Regular 13-gon
    inits.append(_regular_polygon(13, radius=1.0))
    
    # 2. Concentric ring configurations (key insight from inspiration)
    # Mix of outer hull points and inner points to balance triangle areas
    configs = []
    
    # 12 outer + 1 center
    hull_pts = _regular_polygon(12, radius=1.0)
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([hull_pts, center]))
    
    # 11 outer + 2 interior
    hull_pts = _regular_polygon(11, radius=1.0)
    interior1 = np.array([[0.3, 0.0]])
    interior2 = np.array([[-0.15, 0.26]])
    configs.append(np.vstack([hull_pts, interior1, interior2]))
    
    # 10 outer + 3 interior (triangle)
    hull_pts = _regular_polygon(10, radius=1.0)
    interior = _regular_polygon(3, radius=0.3)
    configs.append(np.vstack([hull_pts, interior]))
    
    # 9 outer + 4 interior (square)
    hull_pts = _regular_polygon(9, radius=1.0)
    interior = _regular_polygon(4, radius=0.35)
    configs.append(np.vstack([hull_pts, interior]))
    
    # 8 outer + 5 interior (pentagon)
    hull_pts = _regular_polygon(8, radius=1.0)
    interior = _regular_polygon(5, radius=0.45)
    configs.append(np.vstack([hull_pts, interior]))
    
    # 7 outer + 6 interior (hexagon)
    hull_pts = _regular_polygon(7, radius=1.0)
    interior = _regular_polygon(6, radius=0.5)
    configs.append(np.vstack([hull_pts, interior]))
    
    # More varied concentric configurations
    outer = _regular_polygon(9, radius=1.0)
    inner = _regular_polygon(4, radius=0.4)
    configs.append(np.vstack([outer, inner]))
    
    outer = _regular_polygon(10, radius=1.0)
    inner = _regular_polygon(3, radius=0.3)
    configs.append(np.vstack([outer, inner]))
    
    outer = _regular_polygon(11, radius=1.0)
    inner = _regular_polygon(2, radius=0.25)
    configs.append(np.vstack([outer, inner]))
    
    # Add perturbed regular 13-gon
    rng = np.random.default_rng(42)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    noise = rng.normal(0, 0.05, N)
    perturbed = np.column_stack([np.cos(angles + noise), np.sin(angles + noise)])
    configs.append(perturbed)
    
    inits.extend(configs)
    
    best_score = -np.inf
    best_points = inits[0].copy()  # Fallback
    
    # Main Optimization Loop
    for i, config in enumerate(inits):
        if time.time() - start_time > time_limit - 15:
            break
            
        # Evaluate initial config
        s = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # Phase 1: L-BFGS-B with soft-min surrogate (annealing beta)
        if time.time() - start_time > time_limit - 15:
            break
        
        current = config.copy()
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - start_time > time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_min_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                # Keep the soft-min optimized point for next beta
                current = optimized
            except Exception:
                pass
        
        # Phase 2: Nelder-Mead refinement with hard score
        if time.time() - start_time > time_limit - 15:
            break
        
        if best_points is not None:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                nm_points = res.x.reshape(-1, 2)
                nm_score = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass
    
    # Random restarts with soft-min optimization
    if time.time() < start_time + time_limit - 15:
        np.random.seed(123)
        n_restarts = 8
        for _ in range(n_restarts):
            if time.time() > start_time + time_limit - 15:
                break
            n_outer = np.random.randint(6, 12)
            n_inner = N - n_outer
            outer = _regular_polygon(n_outer, radius=np.random.uniform(0.8, 1.2))
            if n_inner > 0:
                inner = _regular_polygon(n_inner, radius=np.random.uniform(0.2, 0.5))
                config = np.vstack([outer, inner])
            else:
                config = outer
            config = config + np.random.normal(0, 0.1, config.shape)
            
            # Evaluate
            s = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()
            
            # Soft-min optimization
            if time.time() > start_time + time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x: _soft_min_objective(x, 50.0),
                    config.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
            except Exception:
                pass
    
    # Final polishing with tight tolerances
    if time.time() - start_time < time_limit - 5:
        try:
            res2 = opt.minimize(
                _hard_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_points = res2.x.reshape(-1, 2)
            final_score = compute_score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass
    
    # Scale points so that the convex hull area is exactly 1.0
    # This ensures the "normalized to unit area" constraint is met
    area_hull_final = _hull_area(best_points)
    if area_hull_final > 1e-9:
        scale_factor = 1.0 / np.sqrt(area_hull_final)
        best_points = best_points * scale_factor
    
    return best_points
