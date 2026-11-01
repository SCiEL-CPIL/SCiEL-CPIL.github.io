"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq

def _smooth_objective(points_flat, beta=50.0):
    """
    Smooth approximation of (max/min)^2 using log-sum-exp.
    This makes the objective differentiable for gradient-based optimizers.
    """
    points = points_flat.reshape(N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    d = dist_sq[iu]
    
    # Avoid log(0)
    d_safe = np.maximum(d, 1e-15)
    
    # log-sum-exp for max: log(sum(exp(d/beta))) * beta ≈ max(d)
    max_approx = beta * np.log(np.sum(np.exp(d_safe / beta)))
    # For min: -log-sum-exp(-d/beta) * beta ≈ min(d)
    min_approx = -beta * np.log(np.sum(np.exp(-d_safe / beta)))
    
    if min_approx <= 1e-12:
        return 1e10
    
    return (max_approx / min_approx) ** 2

def _objective(points_flat):
    """Objective function for Nelder-Mead. Takes flat array of (N, D) points."""
    points = points_flat.reshape(N, D)
    return _compute_score(points)

def _generate_fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci sphere method."""
    points = np.zeros((n, D))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(n):
        z = 1 - (i + 0.5) * 2.0 / n
        r = np.sqrt(1 - z * z)
        theta = phi * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    return points

def _generate_cuboctahedron_plus_2():
    """Cuboctahedron (12 vertices) + 2 poles."""
    # Cuboctahedron vertices: all permutations of (0, ±1, ±1)
    verts = []
    for a in [0, 1, -1]:
        for b in [0, 1, -1]:
            for c in [0, 1, -1]:
                if abs(a) + abs(b) + abs(c) == 2:
                    verts.append([a, b, c])
    # Deduplicate
    seen = set()
    unique_verts = []
    for v in verts:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_verts.append(v)
    
    pts = np.array(unique_verts, dtype=float)
    # Scale to unit sphere
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Add 2 poles
    poles = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    return np.vstack([pts, poles])

def _generate_rings_symmetric():
    """2 poles + 4 rings of 3 points each at different z-levels."""
    pts = []
    pts.append([0, 0, 1])
    pts.append([0, 0, -1])
    
    # 12 remaining points: 4 rings of 3
    z_levels = [0.75, 0.25, -0.25, -0.75]
    for ring_idx, z in enumerate(z_levels):
        r = np.sqrt(1 - z**2)
        offset = ring_idx * np.pi / 3  # Rotate each ring
        for i in range(3):
            theta = 2 * np.pi * i / 3 + offset
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
    
    return np.array(pts)

def _initialization_candidates():
    """
    Generate a list of initial configurations for the optimizer.
    """
    candidates = []
    
    # 1. Fibonacci Sphere
    candidates.append(_generate_fibonacci_sphere(N))
    
    # 2. Cuboctahedron + 2 poles
    candidates.append(_generate_cuboctahedron_plus_2())
    
    # 3. Symmetric rings
    candidates.append(_generate_rings_symmetric())
    
    # 4. Random Sphere (multiple seeds)
    for seed in range(8):
        rng_i = np.random.default_rng(seed * 12345)
        pts = rng_i.standard_normal((N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
    
    # 5. 2 Poles + 12 on equator (dodecagon)
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(12):
        theta = 2 * np.pi * i / 12
        pts[2+i] = [np.cos(theta), np.sin(theta), 0]
    candidates.append(pts)
    
    # 6. 2 Poles + 6 at z=0.5 + 6 at z=-0.5
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    r_half = np.sqrt(1 - 0.5**2)
    for i in range(6):
        theta = 2 * np.pi * i / 6
        pts[2+i] = [r_half * np.cos(theta), r_half * np.sin(theta), 0.5]
    for i in range(6):
        theta = 2 * np.pi * i / 6 + np.pi/6
        pts[8+i] = [r_half * np.cos(theta), r_half * np.sin(theta), -0.5]
    candidates.append(pts)
    
    return candidates

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses multi-start L-BFGS-B with smooth objective, then polishes with Nelder-Mead.
    """
    best_score = 1e10
    best_points = None
    
    candidates = _initialization_candidates()
    
    # Phase 1: L-BFGS-B with smooth objective from each candidate
    for i, init_pts in enumerate(candidates):
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                _smooth_objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Evaluate with true (non-smooth) objective
            true_score = _compute_score(res.x.reshape(N, D))
            
            if true_score < best_score:
                best_score = true_score
                best_points = res.x.reshape(N, D)
        except Exception as e:
            continue
            
    # Phase 2: Polish best result with Nelder-Mead using true objective
    if best_points is not None:
        x0 = best_points.flatten()
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            true_score = _compute_score(res.x.reshape(N, D))
            if true_score < best_score:
                best_score = true_score
                best_points = res.x.reshape(N, D)
        except Exception:
            pass
    
    # Phase 3: Try a few more Nelder-Mead from other good candidates
    for i, init_pts in enumerate(candidates):
        if i == 0:
            continue
        x0 = init_pts.flatten()
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            true_score = _compute_score(res.x.reshape(N, D))
            if true_score < best_score:
                best_score = true_score
                best_points = res.x.reshape(N, D)
        except Exception:
            continue
            
    # Fallback
    if best_points is None:
        best_points = np.random.randn(N, D)
        
    return best_points
# EVOLVE-BLOCK-END
