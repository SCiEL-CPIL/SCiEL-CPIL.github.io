"""Covering the sphere with 50 equal caps - optimized solver."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        return 180.0
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            # simplex has 3 indices for triangular facets
            if len(simplex) != 3:
                continue
            p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumcenter of the spherical triangle
            # The circumcenter in 3D is the normalized cross product approach
            # or solve the system: c dot p0 = c dot p1 = c dot p2, |c| = 1
            # Using the formula: c = (p0 x p1) x (p1 x p2) normalized... 
            # Actually, the circumcenter of a spherical triangle can be found as:
            # The plane through p0, p1, p2 has normal n = (p1-p0) x (p2-p0)
            # But for unit sphere, the circumcenter is the intersection of the sphere
            # with the plane equidistant from all three points.
            
            # Method: find c such that c·p0 = c·p1 = c·p2 and |c|=1
            # This means c is perpendicular to (p0-p1) and (p0-p2)
            # So c is parallel to (p0-p1) x (p0-p2)
            
            d1 = p0 - p1
            d2 = p0 - p2
            c = np.cross(d1, d2)
            cnorm = np.linalg.norm(c)
            if cnorm < 1e-15:
                continue
            c = c / cnorm
            
            # The angular distance from c to p0 is the circumradius
            dot_val = np.dot(c, p0)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle = np.arccos(dot_val)
            
            # Check that c is on the correct side (the cap should be empty)
            # The circumcenter should be such that the cap centered at c with radius angle
            # contains p0, p1, p2 on its boundary and no other points inside.
            # For now, just track the max.
            if angle > max_radius:
                max_radius = angle
        
        return np.degrees(max_radius)
    except Exception:
        # Fallback: dense sampling
        n_samples = 20000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        dots = sample_points @ points.T
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(angles))


def _fibsphere(n, seed=0):
    """Generate n points on the sphere using Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Small random perturbation
    noise = rng.randn(n, 3) * 0.01
    pts = pts + noise
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    return pts


def _repulsion_relaxation(points, max_iter=200, lr=0.05):
    """Repulsion-based relaxation: push points apart to minimize max distance."""
    points = points.copy()
    
    for it in range(max_iter):
        # Compute pairwise distances
        dots = points @ points.T
        np.fill_diagonal(dots, 2.0)  # Avoid self-distance
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        
        # For each point, find the nearest neighbor distance
        min_dists = np.min(dists, axis=1)
        
        # Gradient: push points away from their nearest neighbors
        grad = np.zeros_like(points)
        for i in range(N):
            j = np.argmin(dists[i])
            if j == i:
                continue
            # Direction to move: away from nearest neighbor
            # The gradient of the angle w.r.t. point i is roughly in the direction
            # of (p_i - p_j) projected to the tangent space
            diff = points[i] - points[j]
            # Project to tangent space at p_i
            tangent = diff - np.dot(diff, points[i]) * points[i]
            tnorm = np.linalg.norm(tangent)
            if tnorm > 1e-10:
                grad[i] += tangent / tnorm
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad, axis=1)
        gnorm = np.maximum(gnorm, 1e-10)
        grad = grad / gnorm[:, np.newaxis]
        
        # Update
        points = points + lr * grad
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay learning rate
        lr *= 0.995
    
    return points


def _lloyd_relaxation(points, max_iter=100, tol=1e-10):
    """Lloyd's algorithm on the sphere using convex hull."""
    points = points.copy()
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(points)
            
            # For each point, find the facets that contain it
            # The Voronoi cell of a point is determined by the facets adjacent to it
            new_points = np.zeros_like(points)
            
            for i in range(N):
                # Find all facets containing point i
                facet_indices = []
                for fidx, simplex in enumerate(hull.simplices):
                    if i in simplex:
                        facet_indices.append(fidx)
                
                if len(facet_indices) == 0:
                    new_points[i] = points[i]
                    continue
                
                # The centroid of the Voronoi cell is the spherical mean of
                # the circumcenters of the adjacent facets
                circumcenters = []
                for fidx in facet_indices:
                    simplex = hull.simplices[fidx]
                    if len(simplex) != 3:
                        continue
                    p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                    d1 = p0 - p1
                    d2 = p0 - p2
                    c = np.cross(d1, d2)
                    cnorm = np.linalg.norm(c)
                    if cnorm > 1e-15:
                        circumcenters.append(c / cnorm)
                
                if len(circumcenters) == 0:
                    new_points[i] = points[i]
                else:
                    centroid = np.mean(circumcenters, axis=0)
                    cnorm = np.linalg.norm(centroid)
                    if cnorm > 1e-10:
                        new_points[i] = centroid / cnorm
                    else:
                        new_points[i] = points[i]
            
            new_points = new_points / np.linalg.norm(new_points, axis=1)[:, np.newaxis]
            
            diff = np.linalg.norm(new_points - points)
            points = new_points
            if diff < tol:
                break
        except Exception:
            break
    
    return points


def _facet_targeted_search(points, max_iter=500):
    """Directly target the worst facet by moving its vertices."""
    points = points.copy()
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(points)
            
            # Find the worst facet (largest circumradius)
            worst_facet = None
            worst_radius = 0.0
            worst_centroid = None
            worst_vertices = None
            
            for simplex in hull.simplices:
                if len(simplex) != 3:
                    continue
                p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                d1 = p0 - p1
                d2 = p0 - p2
                c = np.cross(d1, d2)
                cnorm = np.linalg.norm(c)
                if cnorm < 1e-15:
                    continue
                c = c / cnorm
                dot_val = np.clip(np.dot(c, p0), -1.0, 1.0)
                angle = np.arccos(dot_val)
                
                if angle > worst_radius:
                    worst_radius = angle
                    worst_facet = simplex
                    worst_centroid = c
                    worst_vertices = [simplex[0], simplex[1], simplex[2]]
            
            if worst_facet is None:
                break
            
            # Move each vertex of the worst facet toward the circumcenter
            step_size = 0.01
            for vi in worst_vertices:
                direction = worst_centroid - points[vi]
                # Project to tangent space
                tangent = direction - np.dot(direction, points[vi]) * points[vi]
                tnorm = np.linalg.norm(tangent)
                if tnorm > 1e-10:
                    tangent = tangent / tnorm
                    points[vi] = points[vi] + step_size * tangent
                    points[vi] = points[vi] / np.linalg.norm(points[vi])
            
            # Normalize all points
            points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
            
            # Decay step size
            step_size *= 0.99
            
        except Exception:
            break
    
    return points


def _lbfgs_optimize(points, max_iter=100):
    """Use L-BFGS to minimize the covering radius."""
    def objective(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms < 1e-10):
            return 180.0
        pts = pts / norms[:, np.newaxis]
        return _covering_radius(pts)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        best_pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts / norms[:, np.newaxis]
        return best_pts
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = 180.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Fibonacci sphere with different seeds
    for seed in range(8):
        initial_configs.append(_fibsphere(N, seed=seed))
    
    # Random configurations
    rng = np.random.RandomState(12345)
    for seed in range(8):
        pts = rng.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
        initial_configs.append(pts)
    
    # Try each initial configuration with a pipeline of optimizations
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Stage 1: Repulsion relaxation
        pts1 = _repulsion_relaxation(init_pts.copy(), max_iter=100)
        r1 = _covering_radius(pts1)
        
        # Stage 2: Lloyd relaxation
        pts2 = _lloyd_relaxation(pts1, max_iter=50)
        r2 = _covering_radius(pts2)
        
        # Stage 3: Facet-targeted search
        pts3 = _facet_targeted_search(pts2, max_iter=200)
        r3 = _covering_radius(pts3)
        
        # Pick the best from all stages
        candidates = [(r1, pts1), (r2, pts2), (r3, pts3)]
        best_r, best_p = min(candidates, key=lambda x: x[0])
        
        if best_r < best_radius:
            best_radius = best_r
            best_points = best_p.copy()
            print(f"Config {idx}: radius = {best_r:.6f} degrees")
        
        # If we have time, try L-BFGS from the best point
        if time.time() < deadline - 10.0 and best_r < best_radius + 0.05:
            pts4 = _lbfgs_optimize(best_p, max_iter=50)
            r4 = _covering_radius(pts4)
            if r4 < best_radius:
                best_radius = r4
                best_points = pts4.copy()
                print(f"Config {idx} (L-BFGS): radius = {r4:.6f} degrees")
    
    # Final check
    if best_points is None:
        best_points = _fibsphere(N, seed=0)
    
    # Normalize final result
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points
