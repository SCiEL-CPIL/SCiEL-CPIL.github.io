import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before 150s
    time_limit = 140.0 

    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            pts = points - points.mean(axis=0)
            # Check for degenerate hull
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        
        if min_area < 0:
            return 0.0
        return min_area / area_hull

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def points_from_radii(radii):
        """
        Generate points from 13 radii with fixed angles 2*pi*k/13.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_symmetric(radii):
        """
        Objective function for symmetric optimization.
        Returns negative score to be minimized.
        """
        pts = points_from_radii(radii)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        """
        Objective function for full 26D optimization.
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    # Phase 1: Optimize in the 13-dimensional symmetric subspace
    # Start with regular polygon (radii = 1)
    initial_radii = np.ones(N)
    
    # Try a few different symmetric initializations
    # 1. Regular
    # 2. Slightly perturbed regular (to break symmetry if needed, though symmetry is assumed)
    # 3. Alternating radii (if even N, but for odd N this might not be optimal, still worth trying)
    
    initializations = [
        np.ones(N),
        np.ones(N) + 0.1 * np.sin(2 * np.pi * np.arange(N) / N), # Small radial wave
        np.ones(N) + 0.1 * np.sin(4 * np.pi * np.arange(N) / N), # Higher harmonic
    ]

    for init_rad in initializations:
        if time.time() > start_time + time_limit - 10:
            break
            
        try:
            # Use Powell method as it is derivative-free and robust for small dimensions
            res = opt.minimize(
                objective_symmetric,
                init_rad,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < -best_score:
                best_radii = res.x
                best_pts_sym = points_from_radii(best_radii)
                best_score_sym = compute_score(best_pts_sym)
                
                if best_score_sym > best_score:
                    best_score = best_score_sym
                    best_points = best_pts_sym.copy()
                    
        except Exception:
            continue

    # Phase 2: Polish in the full 26-dimensional space
    # Start from the best symmetric configuration found
    if best_points is not None:
        if time.time() > start_time + time_limit - 5:
            return best_points
            
        try:
            # Nelder-Mead is good for this size
            res = opt.minimize(
                objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = compute_score(candidate_pts)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_pts.copy()
                
        except Exception:
            pass

    # Phase 3: If time permits, try a random restart in full space 
    # to ensure we didn't miss a completely different topology
    if time.time() < start_time + time_limit - 15:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() > start_time + time_limit - 5:
                break
            
            # Random points in a circle
            r = rng.uniform(0.5, 1.0, N)
            theta = rng.uniform(0, 2 * np.pi, N)
            theta.sort() # Sort to maintain some ordering, though not strictly necessary
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            # Quick polish
            try:
                res = opt.minimize(
                    objective_full,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                candidate_pts = res.x.reshape(-1, 2)
                candidate_score = compute_score(candidate_pts)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts.copy()
            except Exception:
                continue

    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon(N, 1.0)

    return best_points
