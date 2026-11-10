import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Fix symmetry: place point 0 at origin, point 1 at (1,0,0) to reduce 42D -> 39D.
    2. Use smooth surrogate (sum of 1/d^2) with L-BFGS-B to get a good starting point.
    3. Polish with Nelder-Mead on the true min/max ratio objective.
    4. Multiple diverse seeds, time-budgeted.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 615  # 15s margin before 630s kill

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        return d_min_sq / d_max_sq

    def score_from_params(params):
        """Reconstruct points from reduced parameter vector and compute score."""
        # params: 39 values for points 2..13 (12 points * 3 coords)
        # Point 0 is at origin, point 1 is at (1,0,0)
        pts = np.zeros((n, d))
        pts[0] = [0.0, 0.0, 0.0]
        pts[1] = [1.0, 0.0, 0.0]
        pts[2:] = params.reshape(n - 2, d)
        return true_score(pts)

    def negative_score(params):
        return -score_from_params(params)

    def smooth_obj(params):
        """Smooth surrogate: minimize sum(1/d_ij^2) over all pairs."""
        pts = np.zeros((n, d))
        pts[0] = [0.0, 0.0, 0.0]
        pts[1] = [1.0, 0.0, 0.0]
        pts[2:] = params.reshape(n - 2, d)
        
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-8)
        return np.sum(1.0 / dists_sq)

    def nm_obj(params):
        """NM objective: maximize true score (minimize negative)."""
        return -score_from_params(params)

    best_score = 0.0
    best_params = None
    best_points = None

    # Generate diverse initial parameter vectors (39-dim)
    initial_params_list = []

    # 1. Random points on sphere, then fix first two
    for trial in range(8):
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        # Fix point 0 to origin, point 1 to (1,0,0) by rotating/translating
        # Simpler: just use pts[2:] as initial params, scale to reasonable range
        p2 = pts[2:].copy()
        # Scale so that max distance is ~2 (point 1 at distance 1 from origin)
        max_d = np.max(np.linalg.norm(p2 - pts[1], axis=1))
        if max_d > 1e-6:
            p2 = p2 * (2.0 / max_d)
        initial_params_list.append(p2.flatten())

    # 2. Icosahedron-based: 12 vertices of icosahedron, add 2 more
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Place icosahedron centered at origin, then fix first two points
    # Use icosahedron vertices for points 0..11, then add 2 more
    # For symmetry breaking: set point 0 to origin, point 1 to (1,0,0)
    # Use icosahedron[2:] for points 2..11, and 2 extra points
    ico_subset = ico_verts[2:12]  # 10 vertices
    # Add 2 more points: opposite-ish directions
    extra1 = np.array([0, 0, 1.0])
    extra2 = np.array([0, 0, -1.0])
    # Scale all to reasonable range
    all_pts = np.vstack([ico_subset, extra1.reshape(1,3), extra2.reshape(1,3)])
    # Scale so max norm is ~2
    max_norm = np.max(np.linalg.norm(all_pts, axis=1))
    if max_norm > 1e-6:
        all_pts = all_pts * (2.0 / max_norm)
    initial_params_list.append(all_pts.flatten())

    # 3. Random perturbation of icosahedron
    for trial in range(4):
        noisy_ico = ico_verts + np.random.randn(12, 3) * 0.1
        noisy_ico = noisy_ico / np.linalg.norm(noisy_ico, axis=1, keepdims=True)
        subset = noisy_ico[2:12]
        extra1 = np.random.randn(3)
        extra1 = extra1 / np.linalg.norm(extra1)
        extra2 = np.random.randn(3)
        extra2 = extra2 / np.linalg.norm(extra2)
        all_pts = np.vstack([subset, extra1.reshape(1,3), extra2.reshape(1,3)])
        max_norm = np.max(np.linalg.norm(all_pts, axis=1))
        if max_norm > 1e-6:
            all_pts = all_pts * (2.0 / max_norm)
        initial_params_list.append(all_pts.flatten())

    # Optimization loop
    for i, init_params in enumerate(initial_params_list):
        if time.time() > deadline - 15:
            break
        
        try:
            # Phase 1: L-BFGS-B on smooth surrogate
            res_smooth = minimize(
                smooth_obj,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            current_params = res_smooth.x
            
            # Phase 2: Nelder-Mead on true objective
            # Use adaptive maxiter based on remaining time
            remaining = deadline - time.time()
            nm_maxiter = int(min(5000, max(1000, remaining * 20)))
            
            res_nm = minimize(
                nm_obj,
                current_params,
                method='Nelder-Mead',
                options={
                    'maxiter': nm_maxiter,
                    'xatol': 1e-8,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            final_params = res_nm.x
            final_pts = np.zeros((n, d))
            final_pts[0] = [0.0, 0.0, 0.0]
            final_pts[1] = [1.0, 0.0, 0.0]
            final_pts[2:] = final_params.reshape(n - 2, d)
            
            score = -res_nm.fun
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_params = final_params.copy()
                best_points = final_pts.copy()
                
        except Exception as e:
            continue

    # Fallback
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p13 = np.array([0, 0, 1.0])
        p14 = np.array([0, 0, -1.0])
        best_points = np.vstack([ico_verts, p13.reshape(1,3), p14.reshape(1,3)])

    # Final validation
    if best_points.shape != (n, d) or not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([[0,0,1.0]]), np.array([[0,0,-1.0]])])

    return best_points
