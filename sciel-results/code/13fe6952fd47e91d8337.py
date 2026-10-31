"""Heilbronn convex regions – 13 points, maximize min triangle area / convex hull area."""
import time
import numpy as np
import warnings
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

# Precompute all C(13,3)=286 triples
TRIPLES = list(combinations(range(N), 3))
TRIPLES_ARR = np.array(TRIPLES, dtype=np.int32)  # shape (286, 3)


def _triangle_areas_vectorized(points):
    """Compute all 286 triangle areas for 13 points. points: (13, 2)"""
    pi = points[TRIPLES_ARR[:, 0]]
    pj = points[TRIPLES_ARR[:, 1]]
    pk = points[TRIPLES_ARR[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: use all points if hull fails (shouldn't happen with 13 distinct pts)
        from scipy.spatial import ConvexHull as CH
        try:
            h = CH(points)
            return h.volume
        except Exception:
            # Shoelace on sorted points
            idx = np.lexsort((points[:, 1], points[:, 0]))
            pts = points[idx]
            x, y = pts[:, 0], pts[:, 1]
            return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = np.min(_triangle_areas_vectorized(points))
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _polar_to_cartesian(polar):
    """Convert (r, theta) pairs to (x, y). polar: (13, 2) -> points: (13, 2)"""
    r = polar[:, 0]
    theta = polar[:, 1]
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _cartesian_to_polar(points):
    """Convert (x, y) to (r, theta)."""
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack([r, theta])


def _logsumexp_smooth_min(areas, beta=50.0):
    """Smooth approximation of min using log-sum-exp.
    As beta -> inf, this approaches -min(areas).
    Returns a smooth surrogate for -min(areas)."""
    # To approximate min: -logsumexp(-beta * areas) / beta
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    if max_val > 700:  # avoid overflow
        return -np.mean(areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum / beta


def _smooth_objective(polar, beta=50.0):
    """Smooth objective in polar coordinates. Minimize this."""
    points = _polar_to_cartesian(polar)
    areas = _triangle_areas_vectorized(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-6:
        return 1.0
    # We want to maximize min_area / hull_area
    # Equivalent to minimizing -min_area / hull_area
    # Use smooth min: smooth_min = -logsumexp(-beta*areas)/beta
    smooth_min = _logsumexp_smooth_min(areas, beta)
    return -smooth_min / hull_area


def _objective_full(points):
    """Objective to minimize: -min_triangle_area / hull_area"""
    areas = _triangle_areas_vectorized(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-6:
        return 1.0
    return -np.min(areas) / hull_area


def _make_ring_config(n_outer, n_inner, r_outer=1.0, r_inner=0.45, angle_offset=0.0):
    """Create a two-ring configuration."""
    angles_outer = 2 * np.pi * np.arange(n_outer) / n_outer
    angles_inner = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner + angle_offset
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([pts_outer, pts_inner])


def _coordinate_descent_polish(points, time_budget, beta_schedule=None):
    """Iterative coordinate descent: optimize each point one at a time.
    For each point, fix others and find best position via local minimization."""
    start = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Adaptive beta schedule for smooth objective
    if beta_schedule is None:
        beta_schedule = [20.0, 30.0, 50.0, 80.0, 120.0, 200.0, 400.0]
    
    n_beta = len(beta_schedule)
    steps_per_beta = max(3, int(time_budget / (n_beta * 0.3)))
    
    for beta_idx, beta in enumerate(beta_schedule):
        for step in range(steps_per_beta):
            if time.time() - start > time_budget:
                break
            
            improved = False
            for i in range(N):
                if time.time() - start > time_budget:
                    break
                
                # Fix all points except i, optimize point i
                fixed_points = np.delete(best_points, i, axis=0)
                
                def _single_point_obj(xi):
                    pt_i = xi.reshape(1, 2)
                    all_pts = np.vstack([best_points[:i], pt_i, best_points[i+1:]])
                    areas = _triangle_areas_vectorized(all_pts)
                    hull_area = _hull_area(all_pts)
                    if hull_area < 1e-6:
                        return 1.0
                    smooth_min = _logsumexp_smooth_min(areas, beta)
                    return -smooth_min / hull_area
                
                x0 = best_points[i].flatten()
                try:
                    res = minimize(_single_point_obj, x0, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                            'adaptive': True})
                    if res.fun < _single_point_obj(x0) - 1e-14:
                        best_points[i] = res.x.reshape(1, 2)
                        improved = True
                except Exception:
                    pass
            
            # Update best score
            current_score = _score(best_points)
            if current_score > best_score:
                best_score = current_score
                improved = True
            
            # Early stop if no improvement for a full pass
            if not improved and beta_idx > 0:
                break
        
        if time.time() - start > time_budget:
            break
    
    return best_points


def _local_polish(points, time_budget):
    """Full local polish using smooth objective with Nelder-Mead in polar coords."""
    start = time.time()
    polar = _cartesian_to_polar(points)
    best_polar = polar.copy()
    best_score = _score(points)
    
    betas = [30.0, 60.0, 100.0, 200.0]
    for beta in betas:
        if time.time() - start > time_budget:
            break
        
        def _polar_obj(p):
            return _smooth_objective(p, beta)
        
        try:
            res = minimize(_polar_obj, best_polar, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < _polar_obj(best_polar) - 1e-14:
                best_polar = res.x
                new_points = _polar_to_cartesian(best_polar)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
        except Exception:
            pass
    
    return _polar_to_cartesian(best_polar)


def _simulated_annealing(points, time_budget):
    """Simulated annealing in Cartesian space."""
    start = time.time()
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.05
    T_min = 0.0001
    cooling = 0.97
    rng = np.random.RandomState(42)
    
    while time.time() - start < time_budget:
        # Propose a move
        idx = rng.randint(N)
        delta = rng.randn(2) * T
        proposed = current.copy()
        proposed[idx] += delta
        
        proposed_score = _score(proposed)
        delta_score = proposed_score - current_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / max(T, 1e-10)):
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    
    # Generate multiple initial configurations
    candidates = []
    
    # 1. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two-ring: 7 outer, 6 inner
    candidates.append(_make_ring_config(7, 6, r_outer=1.0, r_inner=0.45))
    candidates.append(_make_ring_config(7, 6, r_outer=1.0, r_inner=0.40))
    candidates.append(_make_ring_config(7, 6, r_outer=1.0, r_inner=0.50, angle_offset=0.1))
    
    # 3. Two-ring: 5 outer, 8 inner
    candidates.append(_make_ring_config(5, 8, r_outer=1.0, r_inner=0.55))
    candidates.append(_make_ring_config(5, 8, r_outer=1.0, r_inner=0.50))
    
    # 4. Three-ring: 5+4+4
    a1 = 2*np.pi*np.arange(5)/5
    a2 = 2*np.pi*np.arange(4)/4 + np.pi/4
    a3 = 2*np.pi*np.arange(4)/4 + np.pi/4 + 0.2
    r1, r2, r3 = 1.0, 0.6, 0.3
    p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
    p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
    p3 = np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
    candidates.append(np.vstack([p1, p2, p3]))
    
    # 5. Perturbed regular polygon
    rng = np.random.RandomState(123)
    base = candidates[0].copy()
    for trial in range(5):
        pert = rng.randn(N, 2) * 0.08
        candidates.append(base + pert)
    
    # 6. Scaled/perturbed rings
    for r_in in [0.35, 0.42, 0.48, 0.55, 0.62]:
        candidates.append(_make_ring_config(7, 6, r_outer=1.0, r_inner=r_in))
    
    # Evaluate all candidates
    best_points = None
    best_score = -1.0
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    # Phase 1: Coordinate descent polish on best candidate
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 10:
        cd_budget = min(remaining * 0.4, 40.0)
        polished = _coordinate_descent_polish(best_points, cd_budget)
        ps = _score(polished)
        if ps > best_score:
            best_score = ps
            best_points = polished
    
    # Phase 2: Try DE for global exploration if time permits
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 25:
        de_budget = min(remaining * 0.35, 35.0)
        bounds = [(-1.5, 1.5)] * (2 * N)
        de_start = time.time()
        
        def _de_obj(x):
            if time.time() - de_start > de_budget:
                return 1.0
            pts = x.reshape(N, 2)
            return _objective_full(pts)
        
        try:
            result = differential_evolution(
                _de_obj, bounds,
                seed=42,
                maxiter=100,
                popsize=12,
                tol=1e-7,
                mutation=(0.4, 1.0),
                recombination=0.6,
                polish=False,
                init='sobol',
                updating='immediate',
                callback=lambda xk, conv: (time.time() - de_start > de_budget)
            )
            de_points = result.x.reshape(N, 2)
            de_score = _score(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
        except Exception:
            pass
    
    # Phase 3: Local polish with smooth objective
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 8:
        lp_budget = min(remaining * 0.5, 25.0)
        polished2 = _local_polish(best_points, lp_budget)
        ps2 = _score(polished2)
        if ps2 > best_score:
            best_score = ps2
            best_points = polished2
    
    # Phase 4: Final coordinate descent
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 5:
        cd_budget2 = remaining - 1.0
        polished3 = _coordinate_descent_polish(best_points, cd_budget2)
        ps3 = _score(polished3)
        if ps3 > best_score:
            best_score = ps3
            best_points = polished3
    
    # Phase 5: Simulated annealing if time remains
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 5:
        sa_budget = remaining - 1.0
        sa_points = _simulated_annealing(best_points, sa_budget)
        sa_score = _score(sa_points)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Final refinement with tight coordinate descent
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 3:
        final_cd = _coordinate_descent_polish(best_points, remaining - 0.5)
        fs = _score(final_cd)
        if fs > best_score:
            best_points = final_cd
    
    return best_points
