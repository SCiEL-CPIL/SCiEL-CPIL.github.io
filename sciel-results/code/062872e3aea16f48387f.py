import numpy as np
import scipy.optimize as opt
import scipy.spatial as spatial
import time
import itertools

N = 14

# Precompute all triangle indices for vectorized area computation
_TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_N_TRI = len(_TRI_INDICES)

def _hull_area(pts):
    """Compute convex hull area, return 0 if degenerate."""
    try:
        hull = spatial.ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(pts):
    """Vectorized computation of all triangle areas (doubled, i.e., 2*area)."""
    # pts shape: (N, 2)
    # For each triangle (i, j, k), area2 = |det(pts[j]-pts[i], pts[k]-pts[i])|
    i_idx = _TRI_INDICES[:, 0]
    j_idx = _TRI_INDICES[:, 1]
    k_idx = _TRI_INDICES[:, 2]
    
    p1 = pts[i_idx]
    p2 = pts[j_idx]
    p3 = pts[k_idx]
    
    v1 = p2 - p1
    v2 = p3 - p1
    # det = v1[0]*v2[1] - v1[1]*v2[0]
    area2 = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return area2

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 0.0
    areas2 = _all_triangle_areas(pts)
    min_area2 = np.min(areas2)
    # min_area = 0.5 * min_area2, hull_area = hull_a
    # score = (0.5 * min_area2) / hull_a = min_area2 / (2 * hull_a)
    return min_area2 / (2.0 * hull_a)

def _neg_score(pts):
    return -_score(pts)

def _soft_min_objective(pts_flat, beta):
    """Soft minimum of triangle areas, normalized by hull area.
    Lower is better (we minimize this)."""
    pts = pts_flat.reshape((N, 2))
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 1e6
    areas2 = _all_triangle_areas(pts)
    # Use softmin: (1/beta) * log(sum(exp(-beta * a))) approximates min(a)
    # But for large beta, exp underflows. Use the log-sum-exp trick.
    # Actually, let's use a simpler approach: power mean with large p
    # Or just use the hard min for the main objective and soft for gradient-based
    
    # For gradient-based optimization, use a differentiable approximation
    # min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    # To avoid overflow, shift by min
    m = np.min(areas2)
    # soft_min = m - (1/beta) * log(sum(exp(-beta * (areas2 - m))))
    # = m - (1/beta) * log(exp(0) + sum(exp(-beta*(areas2-m))))
    # = m - (1/beta) * log(1 + sum(exp(-beta*(areas2-m))))
    shifted = areas2 - m
    # exp(-beta * shifted) where shifted >= 0, so values <= 1
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_area2 = m - log_sum / beta
    
    return soft_min_area2 / (2.0 * hull_a)

def _initial_regular_polygon():
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])

def _initial_mixed():
    """A mix of regular polygon with slight perturbations."""
    pts = _initial_regular_polygon()
    rng = np.random.default_rng(42)
    # Add small angular perturbations
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    perturbation = rng.normal(0, 0.1, N)
    theta = theta + perturbation
    # Vary radii slightly
    radii = 1.0 + rng.normal(0, 0.05, N)
    return np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])

def _initial_hexagonal():
    """Points arranged in a hexagonal-like pattern."""
    # 3 rings: 1 center, 6 inner, 7 outer (not exact, but a start)
    # Actually for 14 points: center + 6 + 7
    pts = []
    pts.append([0.0, 0.0])
    # Inner ring: 6 points
    for i in range(6):
        theta = i * np.pi / 3
        pts.append([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
    # Outer ring: 7 points
    for i in range(7):
        theta = i * 2 * np.pi / 7
        pts.append([1.0 * np.cos(theta), 1.0 * np.sin(theta)])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = 130.0  # 130s safety margin (limit is 150s)
    
    best_pts = _initial_regular_polygon()
    best_score = _score(best_pts)
    
    def time_check():
        return time.time() - start_time > deadline
    
    # Strategy 1: Run differential evolution with a reasonable budget
    # Use vectorized objective for speed
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    def de_objective(params):
        pts = params.reshape((N, 2))
        hull_a = _hull_area(pts)
        if hull_a < 1e-10:
            return 1e6
        areas2 = _all_triangle_areas(pts)
        min_area2 = np.min(areas2)
        return -min_area2 / (2.0 * hull_a)
    
    # Run DE with limited iterations
    try:
        # Estimate: each eval ~ 0.5ms with vectorized ops
        # DE: popsize=15, maxiter=300 => ~4500 evals => ~2.25s
        # Let's be more aggressive: popsize=20, maxiter=500 => 10000 evals => ~5s
        res = opt.differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=500,
            tol=1e-8,
            popsize=20,
            polish=True,
            mutation=(0.5, 1.0),
            recombination=0.8,
            updating='immediate'
        )
        if not time_check():
            candidate = res.x.reshape((N, 2))
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate
    except Exception:
        pass
    
    # Strategy 2: Gradient-based refinement from multiple starting points
    if not time_check():
        starting_points = [
            _initial_regular_polygon(),
            _initial_mixed(),
            _initial_hexagonal(),
            best_pts.copy()
        ]
        
        # Add some random perturbations of best_pts
        rng = np.random.default_rng(123)
        for _ in range(4):
            perturbed = best_pts + rng.normal(0, 0.1, (N, 2))
            starting_points.append(perturbed)
        
        for init_pts in starting_points:
            if time_check():
                break
            
            init_flat = init_pts.flatten()
            
            # Try with different beta values for soft-min
            for beta in [50.0, 100.0, 200.0]:
                if time_check():
                    break
                
                try:
                    def soft_obj(params):
                        return _soft_min_objective(params, beta)
                    
                    res2 = opt.minimize(
                        soft_obj,
                        init_flat,
                        method='L-BFGS-B',
                        bounds=[(-1.5, 1.5)] * (2 * N),
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if not time_check():
                        candidate = res2.x.reshape((N, 2))
                        s = _score(candidate)
                        if s > best_score:
                            best_score = s
                            best_pts = candidate
                except Exception:
                    pass
    
    # Strategy 3: Local search / Nelder-Mead refinement
    if not time_check():
        try:
            res3 = opt.minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if not time_check():
                candidate = res3.x.reshape((N, 2))
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_pts = candidate
        except Exception:
            pass
    
    # Strategy 4: More DE runs with different seeds if time permits
    if not time_check():
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time > 10:
            try:
                # Run a second DE with different seed and smaller budget
                res4 = opt.differential_evolution(
                    de_objective,
                    bounds,
                    seed=7,
                    maxiter=200,
                    tol=1e-8,
                    popsize=15,
                    polish=True,
                    mutation=(0.3, 0.8),
                    recombination=0.7,
                    updating='immediate'
                )
                if not time_check():
                    candidate = res4.x.reshape((N, 2))
                    s = _score(candidate)
                    if s > best_score:
                        best_score = s
                        best_pts = candidate
            except Exception:
                pass
    
    # Strategy 5: Final polish with L-BFGS-B on hard min (using soft with high beta)
    if not time_check():
        try:
            def hard_obj(params):
                return _soft_min_objective(params, 500.0)
            
            res5 = opt.minimize(
                hard_obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-1.5, 1.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if not time_check():
                candidate = res5.x.reshape((N, 2))
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_pts = candidate
        except Exception:
            pass
    
    # Final check
    if time_check():
        # Return best found so far
        pass
    
    return best_pts
