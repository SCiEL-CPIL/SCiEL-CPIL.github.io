# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # safety margin before 630s limit

    def _min_triangle_area(pts):
        """Compute the minimum triangle area among all C(n,3) triangles."""
        n_pts = len(pts)
        min_area = float('inf')
        for i, j, k in combinations(range(n_pts), 3):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
            if area < min_area:
                min_area = area
        return min_area

    def _hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return float('inf')

    def _score(pts):
        """Compute the ratio A_min / A_hull."""
        try:
            a_min = _min_triangle_area(pts)
            a_hull = _hull_area(pts)
            if a_hull <= 0 or not np.isfinite(a_hull):
                return 0.0
            return a_min / a_hull
        except Exception:
            return 0.0

    def _negative_ratio(pts_flat):
        """Negative score for minimization."""
        pts = pts_flat.reshape(-1, 2)
        return -_score(pts)

    def _soft_min_objective(pts_flat, beta=50.0):
        """
        Soft-min surrogate: -log-sum-exp of triangle areas / hull area.
        This is smoother than the exact min, enabling gradient-based methods.
        """
        pts = pts_flat.reshape(-1, 2)
        n_pts = len(pts)
        
        # Compute all triangle areas
        areas = []
        for i, j, k in combinations(range(n_pts), 3):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
            areas.append(area)
        
        areas = np.array(areas)
        
        # Compute hull area
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 1e10  # Large penalty for invalid configurations
        
        if a_hull <= 0 or not np.isfinite(a_hull):
            return 1e10
        
        # Soft min: -log(sum(exp(-beta * area))) / beta / a_hull
        # This approximates min(areas) / a_hull for large beta
        scaled = -beta * areas / a_hull
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum_exp / beta
        
        return -soft_min  # We want to maximize, so minimize the negative

    def _voronoi_centroid(points, idx):
        """Compute approximate Voronoi cell centroid for point idx using Delaunay."""
        try:
            tri = Delaunay(points)
            neighbors = set()
            for simplex in tri.simplices:
                if idx in simplex:
                    for other in simplex:
                        if other != idx:
                            neighbors.add(other)
            if neighbors:
                neighbor_pts = points[list(neighbors)]
                return neighbor_pts.mean(axis=0)
        except Exception:
            pass
        return points[idx]

    def _lloyd_iteration(points, interior_indices, step=0.1):
        """One Lloyd relaxation step for interior points."""
        new_points = points.copy()
        for idx in interior_indices:
            centroid = _voronoi_centroid(points, idx)
            new_points[idx] = points[idx] + step * (centroid - points[idx])
        return new_points

    def _polish(points, max_iter=200):
        """Polish using coordinate descent on individual coordinates."""
        pts = points.copy()
        current_score = _score(pts)
        
        for it in range(max_iter):
            if time.time() > deadline - 60:
                break
            
            improved = False
            for i in range(n):
                for coord in range(2):
                    # Try small perturbations
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                        trial = pts.copy()
                        trial[i, coord] += delta
                        s = _score(trial)
                        if s > current_score + 1e-10:
                            pts = trial
                            current_score = s
                            improved = True
                            break
            
            if not improved:
                break
        
        return pts, current_score

    # Generate multiple structured initial configurations
    rng = np.random.default_rng(seed=42)
    
    inits = []
    
    # Config 1: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Config 2: Concentric rings (4 inner, 9 outer)
    r1, r2 = 0.35, 0.85
    a_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    a_outer = np.linspace(0, 2 * np.pi, 9, endpoint=False) + np.pi / 9
    inner = np.column_stack([r1 * np.cos(a_inner), r1 * np.sin(a_inner)])
    outer = np.column_stack([r2 * np.cos(a_outer), r2 * np.sin(a_outer)])
    inits.append(np.vstack([inner, outer]))
    
    # Config 3: Concentric rings with different offsets (3 inner, 10 outer)
    r1, r2 = 0.3, 0.8
    a_inner = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    a_outer = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    inner = np.column_stack([r1 * np.cos(a_inner), r1 * np.sin(a_inner)])
    outer = np.column_stack([r2 * np.cos(a_outer), r2 * np.sin(a_outer)])
    inits.append(np.vstack([inner, outer]))
    
    # Config 4: Random normalized
    init4 = rng.random((n, 2))
    init4 = (init4 - init4.mean(axis=0)) / (init4.std(axis=0) + 1e-10)
    inits.append(init4)
    
    # Config 5: Another random
    init5 = rng.random((n, 2))
    init5 = (init5 - init5.mean(axis=0)) / (init5.std(axis=0) + 1e-10)
    inits.append(init5)

    best_points = None
    best_score_val = 0.0

    for init in inits:
        if time.time() > deadline - 120:
            break

        # Lloyd relaxation for interior points
        points = init.copy()
        try:
            hull = ConvexHull(points)
            hull_indices = set(hull.vertices)
            interior_indices = [i for i in range(n) if i not in hull_indices]
        except Exception:
            interior_indices = list(range(n))

        # Run Lloyd relaxation
        for iteration in range(150):
            if time.time() > deadline - 120:
                break
            points = _lloyd_iteration(points, interior_indices, step=0.1)

        # Powell optimization with soft-min surrogate
        x0 = points.flatten()
        
        try:
            result = minimize(_soft_min_objective, x0, method='Powell',
                             options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-6})
            candidate = result.x.reshape(-1, 2)
            s = _score(candidate)
            if s > best_score_val:
                best_score_val = s
                best_points = candidate.copy()
        except Exception:
            pass

        # Polish with coordinate descent
        if best_points is not None and time.time() < deadline - 60:
            polished, polished_score = _polish(best_points, max_iter=100)
            if polished_score > best_score_val:
                best_score_val = polished_score
                best_points = polished.copy()

    # If no valid result, return a regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
