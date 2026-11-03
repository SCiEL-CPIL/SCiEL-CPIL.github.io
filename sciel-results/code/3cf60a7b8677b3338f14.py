"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Combines barycentric parameterization with FPS initialization, SLSQP optimization,
and Nelder-Mead polishing for robust convergence."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian (N, 2) to barycentric (N, 3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    # Solve: x = b1 + 0.5*b2, y = (sqrt(3)/2)*b2
    b2 = 2.0 * y / np.sqrt(3)
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()

def _objective_bary(bary_flat):
    """Objective function: negative min triangle area. Input is flat barycentric coords."""
    bary = bary_flat.reshape(N, 3)
    # Ensure valid barycentric coordinates (clip to simplex)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    return -_min_triangle_area(pts)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """Generate well-spread initial points using Farthest Point Sampling."""
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        # Generate random candidates uniformly in triangle
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        # Calculate min distance from each candidate to existing points
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Pick candidate with max min_dist
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    
    return points

def _generate_symmetric_initialization():
    """Generate a symmetric initial configuration."""
    pts = []
    center = (V0 + V1 + V2) / 3.0
    pts.append(center)
    
    # Ring 1: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.18
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        pts.append([x, y])
    
    # Ring 2: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        r = 0.32
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        pts.append([x, y])
    
    pts = np.array(pts)
    # Clamp to triangle if needed
    bary = _cart_to_bary(pts)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= bary.sum(axis=1, keepdims=True)
    return _bary_to_cart(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary_flat = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initial_bary_list = []
    
    # FPS initializations
    n_fps = 8
    for i in range(n_fps):
        if time_left() < 25.0:
            break
        pts = _generate_fps_initialization(rng)
        bary = _cart_to_bary(pts)
        bary = np.clip(bary, 0.0, 1.0)
        bary /= bary.sum(axis=1, keepdims=True)
        initial_bary_list.append(bary.flatten())
    
    # Symmetric initialization
    if time_left() > 25.0:
        pts_sym = _generate_symmetric_initialization()
        bary_sym = _cart_to_bary(pts_sym)
        bary_sym = np.clip(bary_sym, 0.0, 1.0)
        bary_sym /= bary_sym.sum(axis=1, keepdims=True)
        initial_bary_list.append(bary_sym.flatten())
    
    # Random initializations
    n_random = 5
    for i in range(n_random):
        if time_left() < 20.0:
            break
        bary = rng.uniform(0.05, 0.95, size=(N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        initial_bary_list.append(bary.flatten())
    
    # Optimize each initialization
    bounds = [(0.0, 1.0)] * (3 * N)
    
    for start_bary in initial_bary_list:
        if time_left() < 12.0:
            break
        
        try:
            # SLSQP for smooth optimization
            res = minimize(
                _objective_bary, start_bary, method='SLSQP', bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-9}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_bary_flat = res.x.copy()
        except Exception:
            pass
    
    # Polish best solution with Nelder-Mead
    if best_bary_flat is not None and time_left() > 18.0:
        try:
            res = minimize(
                _objective_bary, best_bary_flat, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_bary_flat = res.x.copy()
        except Exception:
            pass
    
    # Additional refinement from perturbed best solution
    if best_bary_flat is not None and time_left() > 25.0:
        for i in range(4):
            if time_left() < 12.0:
                break
            try:
                noise = rng.normal(0, 0.008, size=3 * N)
                perturbed = np.clip(best_bary_flat + noise, 0.0, 1.0)
                perturbed /= perturbed.reshape(N, 3).sum(axis=1, keepdims=True)
                perturbed = perturbed.flatten()
                
                res = minimize(
                    _objective_bary, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_bary_flat = res.x.copy()
            except Exception:
                pass
    
    # Convert best barycentric to Cartesian
    if best_bary_flat is None:
        # Fallback: use FPS
        pts = _generate_fps_initialization(rng)
    else:
        bary = best_bary_flat.reshape(N, 3)
        bary = np.clip(bary, 0.0, 1.0)
        bary /= bary.sum(axis=1, keepdims=True)
        pts = _bary_to_cart(bary)
    
    return np.array(pts)
