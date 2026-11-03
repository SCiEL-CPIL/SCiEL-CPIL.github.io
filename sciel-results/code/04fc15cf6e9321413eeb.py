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
        
        # Check if Voronoi vertex is valid (inside the spherical triangle)
        # For a valid facet, the dot products should all be positive and < 1
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        if circumradius > max_circ:
            max_circ = circumradius
    
    return max_circ

def _compute_facet_circumradii_and_centers(points):
    """
    For each point, compute the maximum circumradius among its adjacent facets,
    and the direction (toward the Voronoi vertex of the worst facet) to move.
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.ones(N) * np.pi, np.zeros((N, 3))
    
    faces = hull.simplices
    max_circ_per_point = np.zeros(N)
    move_dirs = np.zeros((N, 3))
    
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
        
        for idx, angle in zip(face, angles):
            if angle >= max_circ_per_point[idx]:
                max_circ_per_point[idx] = angle
                move_dirs[idx] = o
    
    norms = np.linalg.norm(move_dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    move_dirs = move_dirs / norms
    
    return max_circ_per_point, move_dirs

def _facet_based_optimize(points, max_iters=200, deadline=None):
    """
    Lloyd-type optimization: move each point toward the Voronoi vertex of its worst facet.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        max_circ, move_dirs = _compute_facet_circumradii_and_centers(points)
        current_radius = np.max(max_circ)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Adaptive step size
        step = 0.1 * np.sqrt(current_radius)
        step = np.clip(step, 0.001, 0.1)
        
        # Move points toward their worst facet Voronoi vertices
        new_points = points + step * move_dirs
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if this improved things; if not, try smaller step
        new_radius = _compute_covering_radius(new_points)
        if new_radius < current_radius:
            points = new_points
        else:
            # Try half step
            new_points = points + 0.5 * step * move_dirs
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius:
                points = new_points
            else:
                # Try quarter step
                new_points = points + 0.25 * step * move_dirs
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                new_radius = _compute_covering_radius(new_points)
                if new_radius < current_radius:
                    points = new_points
                else:
                    break  # No improvement possible with this step
    
    return best_points, best_radius

def _nelder_mead_polish(points, max_iters=50, deadline=None):
    """
    Polish the configuration using Nelder-Mead on the exact covering radius.
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

def _facet_force_optimize(points, max_iters=300, lr=0.01, deadline=None):
    """
    Facet-based force balance optimization.
    Points in large-circumradius facets are pulled toward the facet center;
    points in small-circumradius facets are pushed away to equalize.
    """
    P = points.copy()
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current facet data
        try:
            hull = ConvexHull(P)
        except Exception:
            break
        
        facet_normals = []
        facet_circumradii = []
        facet_points = []
        
        for simplex in hull.simplices:
            verts = P[simplex]
            if len(verts) < 3:
                continue
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            # Ensure normal points outward
            if np.dot(normal, v1) < 0:
                normal = -normal
            
            # Compute circumradius
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            d = np.dot(n, v1)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            
            dots = np.array([np.dot(v1, o), np.dot(v2, o), np.dot(v3, o)])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            circumradius = np.max(angles)
            
            facet_normals.append(normal)
            facet_circumradii.append(circumradius)
            facet_points.append(list(simplex))
        
        if not facet_points:
            break
        
        # Track which facets each point belongs to
        point_facets = [[] for _ in range(N)]
        for idx, pts in enumerate(facet_points):
            for p_idx in pts:
                point_facets[p_idx].append(idx)
        
        # Compute forces on each point
        forces = np.zeros((N, 3))
        avg_cr = np.mean(facet_circumradii)
        
        for i in range(N):
            if not point_facets[i]:
                continue
            
            p = P[i]
            force = np.zeros(3)
            
            for f_idx in point_facets[i]:
                facet_normal = facet_normals[f_idx]
                circumradius = facet_circumradii[f_idx]
                
                # Facet center direction (from origin)
                center_dir = facet_normal / (np.linalg.norm(facet_normal) + 1e-12)
                
                # Normalized deviation: positive if this facet is larger than average
                deviation = circumradius - avg_cr
                
                # Force direction: toward facet center if deviation > 0, away if < 0
                force += deviation * center_dir
            
            # Add a small repulsion from other nearby points to avoid coincident points
            for j in range(N):
                if i == j:
                    continue
                d = np.dot(P[i], P[j])
                if d > 0.999:
                    force -= 0.1 * P[j]
            
            # Project force to tangent plane
            dot_fp = np.dot(force, p)
            force_tangent = force - dot_fp * p
            
            forces[i] = force_tangent
        
        # Update points
        P_new = P + lr * forces
        P_new = P_new / np.linalg.norm(P_new, axis=1, keepdims=True)
        
        # Check for improvements
        new_radius = _compute_covering_radius(P_new)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_P = P_new.copy()
            P = P_new
        else:
            # Adaptive learning rate: reduce if no improvement
            lr *= 0.95
        
        # Early stopping if convergence
        if lr < 1e-6:
            break
    
    return best_P, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Multiple restarts with different initializations
    best_points = None
    best_radius = np.pi
    
    # Initial configurations to try
    inits = [_fibonacci_spiral(N)]
    
    # Add some perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.05
        perturbed = perturbed + noise
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        current_points = init.copy()
        current_radius = _compute_covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
        
        # Phase 1: Facet-based Lloyd optimization
        if time.time() < deadline - 25.0:
            points, radius = _facet_based_optimize(current_points, max_iters=100, deadline=deadline - 25.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 2: Facet force balance optimization
        if time.time() < deadline - 20.0:
            points, radius = _facet_force_optimize(current_points, max_iters=200, lr=0.01, deadline=deadline - 20.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 15.0:
            points, radius = _nelder_mead_polish(current_points, max_iters=50, deadline=deadline - 15.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
    
    # Final polish on the best configuration
    if best_points is not None and time.time() < deadline - 5.0:
        points, radius = _facet_based_optimize(best_points, max_iters=50, deadline=deadline - 5.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
        
        points, radius = _nelder_mead_polish(best_points, max_iters=30, deadline=deadline - 2.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # Ensure output is normalized
    if best_points is not None:
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    else:
        best_points = _fibonacci_spiral(N)
    
    return best_points
