"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_DEADLINE = 140.0  # safety margin before 150s kill

# Precompute all triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _barycentric_to_cartesian(b):
    """Convert barycentric coords (n,3) to cartesian (n,2) in the unit triangle."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # Point = b0*A + b1*B + b2*C
    x = b[:, 1] * 1.0 + b[:, 2] * 0.5
    y = b[:, 2] * _H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert cartesian (n,2) to barycentric coords (n,3)."""
    x, y = pts[:, 0], pts[:, 1]
    b2 = y / _H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _objective_neg_min_area_flat(flat):
    """Objective for SLSQP: negative min triangle area (to minimize)."""
    b = flat.reshape(N, 3)
    # Ensure barycentric coords are valid (non-negative, sum to 1)
    b = np.maximum(b, 0)
    b = b / b.sum(axis=1, keepdims=True)
    pts = _barycentric_to_cartesian(b)
    return -_min_triangle_area(pts)


def _constraints_barycentric(flat):
    """Constraints: barycentric coords sum to 1 for each point."""
    b = flat.reshape(N, 3)
    return b.sum(axis=1) - 1.0


def _initial_symmetric():
    """Create symmetric initial configuration: vertices, edge midpoints, centroid."""
    # Barycentric coords for symmetric points
    # Vertices: (1,0,0), (0,1,0), (0,0,1)
    # Edge midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Centroid: (1/3,1/3,1/3)
    # That's 7 points. Need 4 more - use points along medians
    b = np.array([
        [1.0, 0.0, 0.0],       # vertex A
        [0.0, 1.0, 0.0],       # vertex B
        [0.0, 0.0, 1.0],       # vertex C
        [0.5, 0.5, 0.0],       # midpoint AB
        [0.5, 0.0, 0.5],       # midpoint AC
        [0.0, 0.5, 0.5],       # midpoint BC
        [1/3, 1/3, 1/3],       # centroid
        # 4 more points - place at 1/4 and 3/4 along medians
        [0.75, 0.125, 0.125],  # on median from A
        [0.125, 0.75, 0.125],  # on median from B
        [0.125, 0.125, 0.75],  # on median from C
        [0.375, 0.375, 0.25],  # interior point
    ])
    return b


def _polish(pts, max_iter=500):
    """High-precision polish pass using SLSQP with tight tolerances."""
    b = _cartesian_to_barycentric(pts)
    b = np.maximum(b, 0)
    b = b / b.sum(axis=1, keepdims=True)
    flat = b.flatten()
    
    bounds = [(0, 1)] * (N * 3)
    cons = [{'type': 'eq', 'fun': _constraints_barycentric}]
    
    result = minimize(
        _objective_neg_min_area_flat,
        flat,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
    )
    
    if result.success:
        b_opt = result.x.reshape(N, 3)
        b_opt = np.maximum(b_opt, 0)
        b_opt = b_opt / b_opt.sum(axis=1, keepdims=True)
        return _barycentric_to_cartesian(b_opt)
    return pts


def construct():
    t_start = time.time()
    
    # Start with symmetric configuration
    b_init = _initial_symmetric()
    pts_init = _barycentric_to_cartesian(b_init)
    best_pts = pts_init.copy()
    best_score = _min_triangle_area(best_pts)
    
    # Phase 1: SLSQP optimization from symmetric start
    flat_init = b_init.flatten()
    bounds = [(0, 1)] * (N * 3)
    cons = [{'type': 'eq', 'fun': _constraints_barycentric}]
    
    result = minimize(
        _objective_neg_min_area_flat,
        flat_init,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
    )
    
    if time.time() - t_start > _DEADLINE:
        return best_pts
    
    if result.success:
        b_opt = result.x.reshape(N, 3)
        b_opt = np.maximum(b_opt, 0)
        b_opt = b_opt / b_opt.sum(axis=1, keepdims=True)
        pts_opt = _barycentric_to_cartesian(b_opt)
        score_opt = _min_triangle_area(pts_opt)
        if score_opt > best_score:
            best_pts = pts_opt
            best_score = score_opt
    
    # Phase 2: Polish with tighter tolerances
    if time.time() - t_start < _DEADLINE - 10:
        polished = _polish(best_pts, max_iter=1000)
        score_polished = _min_triangle_area(polished)
        if score_polished > best_score:
            best_pts = polished
            best_score = score_polished
    
    # Phase 3: Try a few perturbed starts if time remains
    if time.time() - t_start < _DEADLINE - 20:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() - t_start > _DEADLINE - 5:
                break
            # Perturb the best solution
            b_current = _cartesian_to_barycentric(best_pts)
            b_current = np.maximum(b_current, 0)
            b_current = b_current / b_current.sum(axis=1, keepdims=True)
            
            # Add small noise
            b_pert = b_current + rng.normal(0, 0.01, size=b_current.shape)
            b_pert = np.maximum(b_pert, 0)
            b_pert = b_pert / b_pert.sum(axis=1, keepdims=True)
            
            flat_pert = b_pert.flatten()
            result_pert = minimize(
                _objective_neg_min_area_flat,
                flat_pert,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            
            if result_pert.success:
                b_opt = result_pert.x.reshape(N, 3)
                b_opt = np.maximum(b_opt, 0)
                b_opt = b_opt / b_opt.sum(axis=1, keepdims=True)
                pts_opt = _barycentric_to_cartesian(b_opt)
                score_opt = _min_triangle_area(pts_opt)
                if score_opt > best_score:
                    best_pts = pts_opt
                    best_score = score_opt
    
    return best_pts
