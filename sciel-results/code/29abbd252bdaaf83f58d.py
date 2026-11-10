import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Use a diverse set of initial configurations (Fibonacci sphere, icosahedron, random).
    2. For each configuration, run a two-phase optimization:
       a. Smooth surrogate optimization (minimize sum of 1/d^2) with L-BFGS-B,
          using RMS normalization to handle scale invariance.
       b. Polish with Nelder-Mead on the true (d_min/d_max)^2 objective.
    3. Keep track of the best result and return it.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 20s margin before 630s kill

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Compute pairwise squared distances
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        return ratio_sq

    def smooth_objective(points):
        """
        Smooth surrogate objective: minimize sum(1/d_ij^2)
        This encourages all points to be far apart.
        """
        dists_sq = pdist(points, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-6)
        return np.sum(1.0 / dists_sq)

    def normalize_rms(points):
        """Normalize points to have unit RMS radius."""
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def fibonacci_sphere(n_points):
        """Generate points on a sphere using the Fibonacci sphere algorithm."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    best_score = 0.0
    best_points = None

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Fibonacci sphere points (good for uniform distribution)
    fib_pts = fibonacci_sphere(n)
    initial_configs.append(fib_pts)
    
    # 2. Random points on sphere
    for _ in range(8):
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= norms
        initial_configs.append(pts)

    # 3. Icosahedron + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts.astype(float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    for _ in range(3):
        p1 = np.random.randn(1, 3)
        p1 /= np.linalg.norm(p1)
        p2 = np.random.randn(1, 3)
        p2 /= np.linalg.norm(p2)
        initial_configs.append(np.vstack([ico_verts, p1, p2]))
    
    # 4. Random points with small perturbation from center
    for _ in range(5):
        pts = np.random.randn(n, d) * 0.5
        initial_configs.append(pts)
    
    # 5. Grid-like starting points
    for _ in range(3):
        pts = np.zeros((n, d))
        indices = np.random.choice(n, n, replace=False)
        for i in range(n):
            pts[i] = np.array([
                np.sin(indices[i] * 0.7),
                np.cos(indices[i] * 1.3),
                np.sin(indices[i] * 2.1)
            ])
        initial_configs.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
            
        try:
            # Phase 1: Smooth Gradient Descent with L-BFGS-B
            def constrained_smooth_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return smooth_objective(pts)

            res_smooth = minimize(
                constrained_smooth_obj, 
                init_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            current_pts = normalize_rms(res_smooth.x.reshape(n, d))
            
            # Phase 2: Nelder-Mead Polish on True Objective
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            # Use larger initial simplex for NM
            initial_simplex = current_pts.flatten().copy()
            
            res_nm = minimize(
                nm_obj, 
                initial_simplex, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            
            final_pts = normalize_rms(res_nm.x.reshape(n, d))
            
            # Calculate score
            score = true_score(final_pts)
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
                
        except Exception as e:
            continue

    # If we have time, do additional random restarts with NM only
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            try:
                # Start from best points with small perturbation
                perturbed = best_points + np.random.randn(n, d) * 0.01
                perturbed = normalize_rms(perturbed)
                
                def nm_obj2(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)
                
                res_nm2 = minimize(
                    nm_obj2, 
                    perturbed.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True}
                )
                
                final_pts2 = normalize_rms(res_nm2.x.reshape(n, d))
                score2 = true_score(final_pts2)
                if score2 > best_score:
                    best_score = score2
                    best_points = final_pts2.copy()
            except Exception:
                continue

    # Fallback if all failed
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts.astype(float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.array([0.0, 0.0, 1.0])
        p2 = np.array([0.0, 0.0, -1.0])
        best_points = np.vstack([ico_verts, p1, p2])

    # Final validation
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts.astype(float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([0.0,0.0,1.0]), np.array([0.0,0.0,-1.0])])

    return best_points
