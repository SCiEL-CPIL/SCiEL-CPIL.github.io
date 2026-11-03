import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """Compute exact covering radius via convex hull facet circumradii (in radians)."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    faces = hull.simplices
    max_circ = 0.0
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        if circumradius > max_circ:
            max_circ = circumradius
    
    return max_circ

def _soft_max_circumradii(points, beta=20.0):
    """
    Compute a smooth approximation to the max circumradius using log-sum-exp.
    Returns (soft_max_value, weights, faces, circumradii, voronoi_vertices).
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.array([]), np.array([], dtype=int), np.array([]), np.array([])
    
    faces = hull.simplices
    circumradii = []
    voronoi_vertices = []
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        circumradii.append(circumradius)
        voronoi_vertices.append(o)
    
    if len(circumradii) == 0:
        return np.pi, np.array([]), faces, np.array([]), np.array([])
    
    circumradii = np.array(circumradii)
    voronoi_vertices = np.array(voronoi_vertices)
    
    # Log-sum-exp for soft max
    max_circ = np.max(circumradii)
    shifted = circumradii - max_circ
    log_sum_exp = max_circ + np.log(np.sum(np.exp(beta * shifted)))
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    return log_sum_exp, weights, faces, circumradii, voronoi_vertices

def _gradient_descent_smooth(points, max_iters=200, deadline=None, beta=20.0):
    """
    Gradient descent using the smooth soft-max objective.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    step = 0.02
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        soft_max, weights, faces, circumradii, voronoi_vertices = _soft_max_circumradii(points, beta)
        
        if len(circumradii) == 0:
            break
        
        current_radius = np.max(circumradii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Compute gradient: for each point, sum weighted contributions from its facets
        # The gradient of the soft-max w.r.t. point positions involves the movement of Voronoi vertices
        # Use a simplified gradient: move each point toward the weighted average of Voronoi vertices of its facets
        total_grad = np.zeros((N, 3))
        
        for k in range(len(circumradii)):
            w = weights[k]
            face = faces[k]
            v = voronoi_vertices[k]
            
            for idx in face:
                # Gradient direction: toward the Voronoi vertex (reduces circumradius)
                # But we need to be careful: moving toward the Voronoi vertex of a facet
                # reduces the distance from that vertex to the facet points
                total_grad[idx] += w * v
        
        # Normalize and project to tangent space
        for i in range(N):
            p = points[i]
            g = total_grad[i]
            # Project to tangent plane
            g_tangent = g - np.dot(g, p) * p
            norm_g = np.linalg.norm(g_tangent)
            if norm_g > 1e-10:
                total_grad[i] = g_tangent / norm_g
        
        # Update points
        new_points = points - step * total_grad
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < current_radius:
            points = new_points
            step *= 1.05
        else:
            step *= 0.9
            if step < 1e-6:
                break
        
        step = np.clip(step, 1e-5, 0.1)
    
    return best_points, best_radius

def _lloyd_iteration(points, deadline=None, max_iters=100):
    """
    Lloyd-type iteration: move each point toward the centroid of its Voronoi cell
    (approximated by the average of adjacent Voronoi vertices).
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        faces = hull.simplices
        
        # For each point, find adjacent facets and their Voronoi vertices
        point_voronoi = [[] for _ in range(N)]
        
        for face in faces:
            a, b, c = points[face[0]], points[face[1]], points[face[2]]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            d = np.dot(n, a)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            
            for idx in face:
                point_voronoi[idx].append(o)
        
        # Move each point toward the average of its adjacent Voronoi vertices
        new_points = points.copy()
        for i in range(N):
            if len(point_voronoi[i]) > 0:
                centroid = np.mean(point_voronoi[i], axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                # Spherical interpolation
                angle = np.arccos(np.clip(np.dot(points[i], centroid), -1.0, 1.0))
                if angle > 1e-10:
                    # Move halfway toward centroid
                    t = 0.5
                    new_points[i] = (np.sin((1-t)*angle) * points[i] + np.sin(t*angle) * centroid) / np.sin(angle)
                    new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
        
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # Try smaller step
            new_points2 = points.copy()
            for i in range(N):
                if len(point_voronoi[i]) > 0:
                    centroid = np.mean(point_voronoi[i], axis=0)
                    centroid = centroid / np.linalg.norm(centroid)
                    angle = np.arccos(np.clip(np.dot(points[i], centroid), -1.0, 1.0))
                    if angle > 1e-10:
                        t = 0.25
                        new_points2[i] = (np.sin((1-t)*angle) * points[i] + np.sin(t*angle) * centroid) / np.sin(angle)
                        new_points2[i] = new_points2[i] / np.linalg.norm(new_points2[i])
            
            new_points2 = new_points2 / np.linalg.norm(new_points2, axis=1, keepdims=True)
            new_radius2 = _compute_covering_radius(new_points2)
            
            if new_radius2 < best_radius:
                best_radius = new_radius2
                best_points = new_points2.copy()
                points = new_points2
            else:
                break
    
    return best_points, best_radius

def _repulsive_force_optimize(points, max_iters=200, deadline=None):
    """
    Optimize using repulsive forces between points plus attraction to equalize facet sizes.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    lr = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute facet data
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        faces = hull.simplices
        facet_circumradii = []
        facet_voronoi = []
        
        for face in faces:
            a, b, c = points[face[0]], points[face[1]], points[face[2]]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            d = np.dot(n, a)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            
            dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            circumradius = np.max(angles)
            
            facet_circumradii.append(circumradius)
            facet_voronoi.append(o)
        
        if len(facet_circumradii) == 0:
            break
        
        facet_circumradii = np.array(facet_circumradii)
        avg_cr = np.mean(facet_circumradii)
        
        # Compute forces
        forces = np.zeros((N, 3))
        
        # Facet-based forces: pull points in large facets toward Voronoi vertex
        point_facets = [[] for _ in range(N)]
        for k, face in enumerate(faces):
            for idx in face:
                point_facets[idx].append(k)
        
        for i in range(N):
            p = points[i]
            force = np.zeros(3)
            
            for k in point_facets[i]:
                cr = facet_circumradii[k]
                v = facet_voronoi[k]
                # If facet is larger than average, pull toward Voronoi vertex
                deviation = cr - avg_cr
                force += deviation * v
            
            # Repulsive force from other points
            for j in range(N):
                if i == j:
                    continue
                d = np.dot(p, points[j])
                if d > 0.95:
                    force -= (d - 0.95) * points[j]
            
            # Project to tangent plane
            force_tangent = force - np.dot(force, p) * p
            forces[i] = force_tangent
        
        # Update
        new_points = points + lr * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            lr = min(lr * 1.02, 0.05)
        else:
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return best_points, best_radius

def _nelder_mead_polish(points, max_iters=100, deadline=None):
    """
    Polish using Nelder-Mead on the exact covering radius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return best_radius
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _compute_covering_radius(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if res.fun < best_radius:
            new_points = res.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    return best_points, best_radius

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _random_init(rng):
    """Generate random points on the sphere."""
    vecs = rng.randn(N, 3)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_radius = np.pi
    
    # Multiple initializations
    inits = [_fibonacci_spiral(N)]
    
    rng = np.random.RandomState(42)
    for _ in range(4):
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.03
        perturbed = perturbed + noise
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        inits.append(perturbed)
    
    # Also try random init
    inits.append(_random_init(rng))
    
    for init in inits:
        if time.time() > deadline - 25.0:
            break
        
        current_points = init.copy()
        current_radius = _compute_covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
        
        # Phase 1: Lloyd iteration
        if time.time() < deadline - 20.0:
            points, radius = _lloyd_iteration(current_points, deadline=deadline - 20.0, max_iters=80)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 2: Gradient descent with smooth objective
        if time.time() < deadline - 15.0:
            points, radius = _gradient_descent_smooth(current_points, max_iters=150, deadline=deadline - 15.0, beta=20.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 3: Repulsive force optimization
        if time.time() < deadline - 10.0:
            points, radius = _repulsive_force_optimize(current_points, max_iters=150, deadline=deadline - 10.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 4: Nelder-Mead polish
        if time.time() < deadline - 5.0:
            points, radius = _nelder_mead_polish(current_points, max_iters=50, deadline=deadline - 5.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 3.0:
        points, radius = _lloyd_iteration(best_points, deadline=deadline - 3.0, max_iters=50)
        if radius < best_radius:
            best_points = points
            best_radius = radius
        
        points, radius = _nelder_mead_polish(best_points, max_iters=30, deadline=deadline - 1.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    if best_points is not None:
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    else:
        best_points = _fibonacci_spiral(N)
    
    return best_points
