import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 130
DEADLINE_MARGIN = 10.0

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _covering_radius_hull(points):
    """
    Compute the covering radius in radians using the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_angle = 0.0
    # Precompute all pairwise dot products? No, N=130 is small, but let's be efficient.
    # For each facet, we need to check if the cap is empty.
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        facet_radius = max(d1, d2, d3)
        
        # Check if empty cap
        dots = points @ normal
        mask = np.ones(N, dtype=bool)
        mask[simplex] = False
        max_dot_other = np.max(dots[mask])
        
        if max_dot_other > np.cos(facet_radius) + 1e-12:
            continue
            
        max_angle = max(max_angle, facet_radius)
        
    return max_angle

def _lloyd_iteration(points, n_iters=50, deadline=None):
    """
    Lloyd's algorithm on the sphere:
    1. Compute Voronoi cells (via Delaunay / Convex Hull)
    2. Move each point to the centroid (spherical mean) of its cell
    """
    points = points.copy().astype(float)
    for _ in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the facet that contains it?
        # Actually, Lloyd's on the sphere uses the Voronoi diagram.
        # The Voronoi cell of point i is the set of points on the sphere closer to i than to any other.
        # The centroid of this cell is the direction of the sum of the normals of the facets adjacent to i?
        # No, it's the spherical centroid of the polygon formed by the circumcenters of the facets sharing vertex i.
        
        # Let's compute the circumcenters of all facets.
        # Then, for each point i, find the facets that share vertex i.
        # The Voronoi cell of i is the convex hull (on the sphere) of the circumcenters of those facets.
        # The centroid is the normalized sum of the circumcenters? No, it's the spherical mean.
        
        # Let's gather circumcenters.
        circumcenters = []
        facet_vertices = []
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            
            circumcenters.append(normal)
            facet_vertices.append(simplex)
        
        if len(circumcenters) == 0:
            break
            
        circumcenters = np.array(circumcenters)
        
        new_points = np.zeros_like(points)
        
        # For each point i, find which facets it belongs to
        for i in range(N):
            # Find facets containing i
            mask = [idx for idx, simplex in enumerate(facet_vertices) if i in simplex]
            if len(mask) == 0:
                new_points[i] = points[i]
                continue
            
            # Get the circumcenters of these facets
            cc = circumcenters[mask]
            
            # Spherical mean: normalize the sum
            mean = np.sum(cc, axis=0)
            norm = np.linalg.norm(mean)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = mean / norm
        
        # Update points
        points = new_points
        
    return points

def _gradient_optimize(points, deadline=None):
    """
    Optimize using gradient descent on a smooth approximation of the covering radius.
    """
    # Use a grid for the objective
    n_theta = 180
    n_phi = 360
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    beta = 20.0
    
    def objective_and_grad(flat_params):
        p = flat_params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        
        dots = grid_points @ p.T
        dots = np.clip(dots, -1.0, 1.0)
        
        # Softmax weights
        gamma = 10.0
        exp_dots = np.exp(gamma * dots)
        sum_exp_dots = exp_dots.sum(axis=1, keepdims=True)
        weights = exp_dots / sum_exp_dots
        
        M_i = (weights * dots).sum(axis=1)
        M_i = np.clip(M_i, -1.0, 1.0)
        M_i_safe = np.clip(M_i, -0.999999, 0.999999)
        d_i = np.arccos(M_i_safe)
        
        obj = np.sum(np.exp(beta * d_i))
        
        exp_d = np.exp(beta * d_i)
        sum_exp_d = exp_d.sum()
        if sum_exp_d == 0:
            return obj, np.zeros_like(flat_params)
        w_i = exp_d / sum_exp_d
        
        W = weights.T @ w_i
        G = weights.T @ (w_i[:, None] * grid_points)
        
        forces = G - W[:, None] * p
        
        radial = (forces * p).sum(axis=1, keepdims=True) * p
        tangent_forces = forces - radial
        
        grad = -tangent_forces
        return obj, grad.ravel()
    
    x0 = points.ravel()
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-5, 'maxfun': 200}
        )
        if res.success or res.fun < objective_and_grad(x0)[0]:
            new_points = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            return new_points
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    inits = []
    inits.append(_fibonacci_sphere(N, offset=0.5))
    inits.append(_fibonacci_sphere(N, offset=0.1))
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_points = None
    best_radius = np.pi
    
    # Evaluate initial configurations
    for pts in inits:
        r = _covering_radius_hull(pts)
        if r < best_radius:
            best_radius = r
            best_points = pts.copy()
        if time.time() > deadline - 10:
            break
            
    # Optimize using Lloyd's algorithm
    if best_points is not None and time.time() < deadline - 5:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=100, deadline=deadline - 5)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
            
    # Try gradient optimization
    if best_points is not None and time.time() < deadline - 10:
        grad_pts = _gradient_optimize(best_points, deadline=deadline - 5)
        r = _covering_radius_hull(grad_pts)
        if r < best_radius:
            best_radius = r
            best_points = grad_pts.copy()
            
    # Try optimizing from other initial configs
    for pts in inits:
        if time.time() > deadline - 10:
            break
        if np.allclose(pts, best_points):
            continue
            
        lloyd_pts = _lloyd_iteration(pts, n_iters=50, deadline=deadline - 5)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
            
            if time.time() < deadline - 10:
                grad_pts = _gradient_optimize(best_points, deadline=deadline - 5)
                r = _covering_radius_hull(grad_pts)
                if r < best_radius:
                    best_radius = r
                    best_points = grad_pts.copy()
                    
    # Final refinement
    if best_points is not None and time.time() < deadline - 2:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=20, deadline=deadline - 1)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_points = lloyd_pts.copy()
            
        if time.time() < deadline - 1:
            grad_pts = _gradient_optimize(best_points, deadline=deadline - 0.5)
            r = _covering_radius_hull(grad_pts)
            if r < best_radius:
                best_points = grad_pts.copy()

    if best_points is None:
        best_points = inits[0]
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    r = _covering_radius_hull(pts)
    print(f"Covering radius: {np.degrees(r):.6f} degrees")
