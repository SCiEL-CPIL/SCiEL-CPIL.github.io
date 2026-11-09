"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    points = np.asarray(points)
    n = len(points)
    if n < 4:
        return np.pi
    
    # Use convex hull to find facets (Voronoi cells on sphere)
    # The covering radius is the max over all facets of the circumradius
    # A facet's circumradius = angular distance from facet center to its vertices
    
    # For efficiency, use the fact that the farthest point from all sites
    # is at the circumcenter of some Delaunay triangle (facet of convex hull)
    
    # Compute convex hull of points on sphere
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    # For each facet, compute the circumcenter and its distance to vertices
    max_r = 0.0
    for simplex in hull.simplices:
        # simplex has 3 indices (triangular facet on sphere)
        if len(simplex) != 3:
            continue
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Circumcenter of triangle on unit sphere
        # The circumcenter is in the plane of the triangle, equidistant from a, b, c
        # For a spherical triangle, the circumcenter (in 3D) can be found by:
        # C = (a + b + c) / |a + b + c| if the triangle is acute
        # But more robustly: solve for point equidistant from a, b, c on sphere
        
        # The circumcenter of a spherical triangle lies along the normal to the plane
        # of the triangle (for the minor arc case)
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            continue
        normal = normal / normal_norm
        
        # The circumcenter is along the normal direction (or opposite)
        # Choose the direction that gives the smaller angular distance to vertices
        c1 = normal
        c2 = -normal
        
        # Compute angular distance from c1 to a
        d1 = np.arccos(np.clip(np.dot(c1, a), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(c2, a), -1.0, 1.0))
        
        # Use the closer one
        if d1 <= d2:
            center = c1
            r = d1
        else:
            center = c2
            r = d2
        
        if r > max_r:
            max_r = r
    
    return max_r


def _covering_radius_and_gradient(points, max_iters=1000, tol=1e-10):
    """
    Compute covering radius and gradient using the critical points approach.
    The covering radius is determined by the circumcenters of the largest Voronoi cells.
    We find these critical points and compute the gradient.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Find the covering radius and identify which facets determine it
    # Use convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    # Compute circumradius for each facet
    facet_radii = np.zeros(len(hull.simplices))
    facet_centers = []
    
    for i, simplex in enumerate(hull.simplices):
        if len(simplex) != 3:
            facet_radii[i] = 0
            facet_centers.append(np.zeros(3))
            continue
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            facet_radii[i] = 0
            facet_centers.append(np.zeros(3))
            continue
        normal = normal / normal_norm
        
        c1 = normal
        c2 = -normal
        
        d1 = np.arccos(np.clip(np.dot(c1, a), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(c2, a), -1.0, 1.0))
        
        if d1 <= d2:
            center = c1
            r = d1
        else:
            center = c2
            r = d2
        
        facet_radii[i] = r
        facet_centers.append(center)
    
    facet_centers = np.array(facet_centers)
    max_r = np.max(facet_radii)
    
    # Find facets that achieve (or are close to) the max radius
    # These are the "tight" constraints
    threshold = max_r - 1e-6
    active = np.where(facet_radii >= threshold)[0]
    
    if len(active) == 0:
        return max_r, np.zeros_like(points)
    
    # For each active facet, compute the gradient contribution
    # The gradient of the circumradius with respect to the points
    # For a facet with circumcenter C and vertices a, b, c:
    # d(r)/d(a) = -C / sin(r) * (projection onto tangent plane at a)
    # More precisely: the gradient of arccos(C·a) w.r.t. a is -C/sqrt(1-(C·a)^2)
    # but projected to the tangent space
    
    grad = np.zeros_like(points)
    
    for idx in active:
        simplex = hull.simplices[idx]
        if len(simplex) != 3:
            continue
        center = facet_centers[idx]
        r = facet_radii[idx]
        
        if r < 1e-10:
            continue
        
        # Gradient of arccos(center · p) w.r.t. p = -center / sqrt(1 - (center·p)^2)
        # But we need to project to tangent space
        for j, pi in enumerate(simplex):
            p = points[pi]
            dot_cp = np.dot(center, p)
            dot_cp = np.clip(dot_cp, -1.0 + 1e-15, 1.0 - 1e-15)
            denom = np.sqrt(1.0 - dot_cp * dot_cp)
            if denom < 1e-15:
                continue
            # Gradient in 3D space
            g3d = -center / denom
            # Project to tangent space at p: g_tangent = g3d - (g3d·p)*p
            proj = np.dot(g3d, p)
            g_tangent = g3d - proj * p
            grad[pi] += g_tangent / len(active)
    
    return max_r, grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_r = _covering_radius(best_points)
    
    # Iterative optimization: find critical points and adjust
    # Use a simple gradient descent with adaptive step size
    
    lr = 0.01
    patience = 0
    max_patience = 50
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute covering radius and gradient
        current_r, grad = _covering_radius_and_gradient(points)
        
        # Update best
        if current_r < best_r - 1e-10:
            best_r = current_r
            best_points = points.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            # Restart with smaller learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
            patience = 0
            # Add small random perturbation
            points = points + 0.01 * np.random.randn(*points.shape)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            continue
        
        # Gradient step
        points = points - lr * grad
        
        # Project back to sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check for coincident points
        min_dist = np.min(np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2))
        if min_dist < 1e-6:
            # Perturb coincident points
            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(points[i] - points[j]) < 1e-6:
                        points[j] += 0.01 * np.random.randn(3)
                        points[j] /= np.linalg.norm(points[j])
    
    # Final polish: use scipy minimize on the covering radius
    if time.time() < deadline - 20:
        def objective(x):
            pts = x.reshape(-1, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            r, _ = _covering_radius_and_gradient(pts)
            return r
        
        def jac(x):
            pts = x.reshape(-1, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            _, g = _covering_radius_and_gradient(pts)
            return g.ravel()
        
        try:
            result = minimize(
                objective, 
                best_points.ravel(), 
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun < best_r:
                best_r = result.fun
                best_points = result.x.reshape(-1, 3)
                best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        except Exception:
            pass
    
    # Ensure no coincident points and valid configuration
    # Check for points in closed hemisphere
    # The configuration should span the sphere
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
