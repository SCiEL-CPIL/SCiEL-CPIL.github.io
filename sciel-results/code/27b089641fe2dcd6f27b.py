"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _spherical_circumcenter(v1, v2, v3):
    """
    Compute the circumcenter and circumradius of a spherical triangle on the unit sphere.
    Returns (circumcenter_unit_vector, circumradius_in_radians).
    """
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d

    A = np.array([
        [e1.dot(v2) - e1.dot(v1), e2.dot(v2) - e2.dot(v1)],
        [e1.dot(v3) - e1.dot(v1), e2.dot(v3) - e2.dot(v1)]
    ])
    b = np.array([
        1.0 - v1.dot(v2),
        1.0 - v1.dot(v3)
    ])

    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d

    C = v1 + uv[0]*e1 + uv[1]*e2
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-12:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
    C = C / C_norm

    dot = np.clip(np.dot(C, v1), -1.0, 1.0)
    r = np.arccos(dot)
    return C, r


def _compute_all_circumradii(points, hull):
    """Compute circumradii and centers for all facets."""
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for i in range(n_facets):
        f = simplices[i]
        C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
        radii[i] = r
        centers[i] = C
    
    return radii, centers


def _covering_radius(points):
    """Compute the covering radius (max circumradius) in radians."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    radii, _ = _compute_all_circumradii(points, hull)
    return np.max(radii)


def _slerp(v1, v2, t):
    """Spherical linear interpolation between two unit vectors."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-10:
        return v1.copy()
    s1 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s2 = np.sin(t * theta) / np.sin(theta)
    result = s1 * v1 + s2 * v2
    norm = np.linalg.norm(result)
    if norm < 1e-15:
        return v1.copy()
    return result / norm


def _lloyd_step(points):
    """One step of Lloyd's algorithm on the sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    point_facets = [[] for _ in range(N)]
    for f in hull.simplices:
        for idx in f:
            point_facets[idx].append(f)
    
    new_points = np.zeros_like(points)
    for i in range(N):
        if not point_facets[i]:
            new_points[i] = points[i]
            continue
        centroid = np.zeros(3)
        count = 0
        for f in point_facets[i]:
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            centroid += C
            count += 1
        if count > 0:
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    return new_points


def _geodesic_bisector_refine(points, max_iterations=500, time_budget=50.0):
    """
    Geodesic-bisector refinement: move vertices of worst facets along geodesics
    toward their circumcenters with adaptive learning rate.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Adaptive learning rate
    lr = 0.15
    lr_min = 0.001
    lr_decay = 0.97
    no_improve_count = 0
    patience = 30
    
    for it in range(max_iterations):
        if time.time() - start > time_budget:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        radii, centers = _compute_all_circumradii(points, hull)
        max_r = np.max(radii)
        
        # Smooth surrogate: use log-sum-exp to weight facets
        # This gives more weight to the truly worst facets
        scaled = radii * 50.0  # temperature parameter
        max_scaled = np.max(scaled)
        weights = np.exp(scaled - max_scaled)
        weights /= np.sum(weights)
        
        # Select top-k worst facets (k adapts to how many are near max)
        sorted_idx = np.argsort(-radii)
        k = max(1, int(np.sum(radii > 0.98 * max_r)))
        k = min(k, 10)  # Cap at 10 facets
        top_facets = sorted_idx[:k]
        
        # Build displacement for each vertex
        displacements = np.zeros_like(points)
        for fi in top_facets:
            w = weights[fi]
            f = hull.simplices[fi]
            C = centers[fi]
            for idx in f:
                v = points[idx]
                # Geodesic direction toward circumcenter
                dot_vc = np.clip(np.dot(v, C), -1.0, 1.0)
                theta = np.arccos(dot_vc)
                if theta > 1e-10:
                    # Tangent direction from v toward C
                    tangent = C - dot_vc * v
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-15:
                        tangent = tangent / t_norm
                        displacements[idx] += w * lr * tangent
                else:
                    displacements[idx] += w * lr * 0.01 * v
        
        # Apply displacements with projection back to sphere
        new_points = points + displacements
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / np.maximum(norms, 1e-15)
        
        # Check improvement
        new_radius = _covering_radius(new_points)
        if new_radius < best_radius - 1e-10:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Slightly increase lr if improving well
            if it < 50:
                lr = min(0.2, lr * 1.02)
        else:
            no_improve_count += 1
            # Decay learning rate
            lr = max(lr_min, lr * lr_decay)
            # If stuck, try a small perturbation
            if no_improve_count > patience:
                # Perturb worst facet vertices slightly
                worst_idx = np.argmax(radii)
                f = hull.simplices[worst_idx]
                for idx in f:
                    noise = np.random.randn(3) * 0.01
                    new_v = points[idx] + noise
                    new_v = new_v / np.linalg.norm(new_v)
                    points[idx] = new_v
                no_improve_count = 0
                lr = max(lr_min, lr * 0.5)
    
    return best_points


def _exact_local_solve(points, time_budget=10.0):
    """
    Exact local optimization on the worst facet using scipy minimize.
    Parametrize the 3 vertices of the worst facet and minimize the max circumradius
    of all facets adjacent to those vertices.
    """
    start = time.time()
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    radii, centers = _compute_all_circumradii(points, hull)
    max_r = np.max(radii)
    
    if max_r < np.radians(18.35):
        return points
    
    # Find worst facet
    worst_idx = np.argmax(radii)
    f = hull.simplices[worst_idx]
    v0, v1, v2 = points[f[0]], points[f[1]], points[f[2]]
    
    # Get all facets that share any of these 3 vertices
    relevant_facets = set()
    for fi in range(len(hull.simplices)):
        sf = hull.simplices[fi]
        for idx in sf:
            if idx in f:
                relevant_facets.add(fi)
                break
    
    relevant_facets = list(relevant_facets)
    
    def _objective(params):
        """Objective: max circumradius among relevant facets."""
        if time.time() - start > time_budget:
            return 1e10
        # Unpack 3 unit vectors (6 params: 2 angles each)
        try:
            p0 = np.array([np.sin(params[0])*np.cos(params[1]),
                          np.sin(params[0])*np.sin(params[1]),
                          np.cos(params[0])])
            p1 = np.array([np.sin(params[2])*np.cos(params[3]),
                          np.sin(params[2])*np.sin(params[3]),
                          np.cos(params[2])])
            p2 = np.array([np.sin(params[4])*np.cos(params[5]),
                          np.sin(params[4])*np.sin(params[5]),
                          np.cos(params[4])])
        except:
            return 1e10
        
        # Check if points are too close
        if np.linalg.norm(p0 - p1) < 0.01 or np.linalg.norm(p1 - p2) < 0.01 or np.linalg.norm(p0 - p2) < 0.01:
            return 1e10
        
        # Build modified points
        mod_points = points.copy()
        mod_points[f[0]] = p0
        mod_points[f[1]] = p1
        mod_points[f[2]] = p2
        
        # Recompute hull
        try:
            new_hull = ConvexHull(mod_points)
        except:
            return 1e10
        
        # Compute max radius among relevant facets
        max_rad = 0.0
        for fi in relevant_facets:
            sf = new_hull.simplices[fi]
            C, r = _spherical_circumcenter(mod_points[sf[0]], mod_points[sf[1]], mod_points[sf[2]])
            max_rad = max(max_rad, r)
        
        return max_rad
    
    # Initial parameters from current positions
    def _vec_to_angles(v):
        v = v / np.linalg.norm(v)
        theta = np.arccos(np.clip(v[2], -1.0, 1.0))
        phi = np.arctan2(v[1], v[0])
        return theta, phi
    
    x0 = np.array([_vec_to_angles(v0) + _vec_to_angles(v1) + _vec_to_angles(v2)])
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    except Exception:
        return points
    
    if result.fun < max_r:
        # Apply the solution
        try:
            p0 = np.array([np.sin(result.x[0])*np.cos(result.x[1]),
                          np.sin(result.x[0])*np.sin(result.x[1]),
                          np.cos(result.x[0])])
            p1 = np.array([np.sin(result.x[2])*np.cos(result.x[3]),
                          np.sin(result.x[2])*np.sin(result.x[3]),
                          np.cos(result.x[2])])
            p2 = np.array([np.sin(result.x[4])*np.cos(result.x[5]),
                          np.sin(result.x[4])*np.sin(result.x[5]),
                          np.cos(result.x[4])])
            points[f[0]] = p0
            points[f[1]] = p1
            points[f[2]] = p2
        except:
            pass
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm for uniform distribution (shorter, since we'll refine more)
    lloyd_deadline = deadline - 60.0
    for _ in range(150):
        if time.time() > lloyd_deadline:
            break
        points = _lloyd_step(points)
    
    # 3. Geodesic-bisector refinement with adaptive learning rate
    refine_time = min(50.0, deadline - time.time())
    points = _geodesic_bisector_refine(points, max_iterations=500, time_budget=refine_time)
    
    # 4. Exact local solve on worst facet
    solve_time = min(15.0, deadline - time.time())
    if solve_time > 5.0:
        points = _exact_local_solve(points, time_budget=solve_time)
    
    # 5. Final geodesic polish with very small steps
    polish_deadline = deadline - 3.0
    for _ in range(200):
        if time.time() > polish_deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        radii, centers = _compute_all_circumradii(points, hull)
        max_r = np.max(radii)
        
        if max_r < np.radians(18.31):
            break
        
        # Move worst facet vertices with tiny geodesic steps
        worst_idx = np.argmax(radii)
        f = hull.simplices[worst_idx]
        C = centers[worst_idx]
        
        step = 0.005
        for idx in f:
            v = points[idx]
            # Slerp toward circumcenter
            new_v = _slerp(v, C, step)
            points[idx] = new_v
    
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-15)
    
    return points
# EVOLVE-BLOCK-END
