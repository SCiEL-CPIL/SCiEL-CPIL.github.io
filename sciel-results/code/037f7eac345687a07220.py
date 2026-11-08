"""Optimized solution for Heilbronn convex regions - 13 points.
Combines vectorized triangle computation with multi-shell (2 and 3 shell) 
parameterized optimization and full-coordinate Nelder-Mead refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=80.0):
    """Negative smooth min area / hull area for gradient-based optimization."""
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Soft min via log-sum-exp
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    
    return -soft_min / hull_area

def _make_shell(n, r, phase):
    """Create n points on a circle of radius r starting at angle phase."""
    angles = phase + 2 * np.pi * np.arange(n) / n
    return np.column_stack((r * np.cos(angles), r * np.sin(angles)))

def _two_shell_to_points(params, n1, n2):
    """params = [r1, r2, phase1, phase2] -> points"""
    r1, r2, ph1, ph2 = params
    p1 = _make_shell(n1, r1, ph1)
    p2 = _make_shell(n2, r2, ph2)
    return np.vstack([p1, p2])

def _three_shell_to_points(params, n1, n2, n3):
    """params = [r1, r2, r3, phase1, phase2, phase3] -> points"""
    r1, r2, r3, ph1, ph2, ph3 = params
    p1 = _make_shell(n1, r1, ph1)
    p2 = _make_shell(n2, r2, ph2)
    p3 = _make_shell(n3, r3, ph3)
    return np.vstack([p1, p2, p3])

def _neg_score_two_shell(params, n1, n2, beta=80.0):
    """Negative score for two-shell parameterization."""
    pts = _two_shell_to_points(params, n1, n2)
    # Center at origin
    pts = pts - np.mean(pts, axis=0)
    # Normalize scale so hull area ~ 1 (or just use as-is, score normalizes)
    areas = _all_triangle_areas(pts)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    hull_area = _hull_area(pts)
    if hull_area <= 1e-10:
        return 1e10
    return -soft_min / hull_area

def _neg_score_three_shell(params, n1, n2, n3, beta=80.0):
    """Negative score for three-shell parameterization."""
    pts = _three_shell_to_points(params, n1, n2, n3)
    pts = pts - np.mean(pts, axis=0)
    areas = _all_triangle_areas(pts)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    hull_area = _hull_area(pts)
    if hull_area <= 1e-10:
        return 1e10
    return -soft_min / hull_area

def _neg_score_full(params, beta=80.0):
    """Negative score for full coordinate optimization."""
    return _soft_min_objective(params, beta)

def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin before 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    def _update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s
    
    # === Phase 1: Evaluate structured initial configurations ===
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    _update_best(regular)
    
    # Two-shell configurations: various splits
    two_shell_splits = [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
    for n1, n2 in two_shell_splits:
        for r1 in [0.2, 0.35, 0.5, 0.65]:
            for r2 in [0.85, 1.0, 1.15]:
                for ph1 in [0.0, np.pi/n1]:
                    for ph2 in [0.0, np.pi/n2]:
                        if time.time() > deadline - 10:
                            break
                        pts = _two_shell_to_points([r1, r2, ph1, ph2], n1, n2)
                        _update_best(pts)
    
    # Three-shell configurations
    three_shell_splits = [(1,2,10), (1,3,9), (2,2,9), (2,3,8), (3,3,7), (3,4,6), (4,4,5)]
    for n1, n2, n3 in three_shell_splits:
        for r1 in [0.15, 0.3]:
            for r2 in [0.5, 0.7]:
                for r3 in [0.95, 1.1]:
                    for ph1 in [0.0]:
                        for ph2 in [0.0, np.pi/n2]:
                            for ph3 in [0.0, np.pi/n3]:
                                if time.time() > deadline - 10:
                                    break
                                pts = _three_shell_to_points([r1, r2, r3, ph1, ph2, ph3], n1, n2, n3)
                                _update_best(pts)
    
    # === Phase 2: Parameterized gradient optimization on shells ===
    # Two-shell optimization
    for n1, n2 in two_shell_splits:
        if time.time() > deadline - 30:
            break
        # Multiple starting points
        starts_2s = []
        for r1 in [0.3, 0.5, 0.7]:
            for r2 in [0.9, 1.1]:
                for ph1 in [0.0, np.pi/n1]:
                    for ph2 in [0.0, np.pi/n2]:
                        starts_2s.append([r1, r2, ph1, ph2])
        
        for start in starts_2s[:8]:  # Limit starts per split
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(
                    _neg_score_two_shell,
                    np.array(start, dtype=float),
                    args=(n1, n2, 60.0),
                    method='L-BFGS-B',
                    bounds=[(0.05, 1.5), (0.05, 1.5), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.x is not None:
                    pts = _two_shell_to_points(res.x, n1, n2)
                    pts = pts - np.mean(pts, axis=0)
                    _update_best(pts)
            except Exception:
                pass
    
    # Three-shell optimization
    for n1, n2, n3 in three_shell_splits:
        if time.time() > deadline - 20:
            break
        starts_3s = []
        for r1 in [0.2, 0.4]:
            for r2 in [0.6, 0.8]:
                for r3 in [1.0, 1.2]:
                    for ph2 in [0.0, np.pi/n2]:
                        for ph3 in [0.0, np.pi/n3]:
                            starts_3s.append([r1, r2, r3, 0.0, ph2, ph3])
        
        for start in starts_3s[:10]:  # Limit starts per split
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(
                    _neg_score_three_shell,
                    np.array(start, dtype=float),
                    args=(n1, n2, n3, 60.0),
                    method='L-BFGS-B',
                    bounds=[(0.05, 1.5), (0.05, 1.5), (0.05, 1.5),
                            (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.x is not None:
                    pts = _three_shell_to_points(res.x, n1, n2, n3)
                    pts = pts - np.mean(pts, axis=0)
                    _update_best(pts)
            except Exception:
                pass
    
    # === Phase 3: Full-coordinate Nelder-Mead refinement from best ===
    if time.time() < deadline - 10 and best_pts is not None:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _neg_score_full,
                x0,
                args=(50.0,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None:
                pts = res.x.reshape(N, 2)
                _update_best(pts)
        except Exception:
            pass
    
    # === Phase 4: Additional Nelder-Mead from shell-optimized results ===
    if time.time() < deadline - 5 and best_pts is not None:
        # Try a few more NM starts from perturbed versions of best
        rng = np.random.default_rng(42)
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            x0 = best_pts.flatten() + rng.normal(0, 0.05, size=N*2)
            try:
                res = minimize(
                    _neg_score_full,
                    x0,
                    args=(40.0,),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, 2)
                    _update_best(pts)
            except Exception:
                pass
    
    # Fallback
    if best_pts is None:
        best_pts = regular
    
    return best_pts
