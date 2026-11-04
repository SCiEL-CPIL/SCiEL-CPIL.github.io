"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and circumradius for each facet.
    Returns:
        hull: ConvexHull object or None
        radii: array of circumradii for each facet
        max_radius: maximum circumradius
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, np.pi
        
    radii = np.zeros(len(hull.simplices))
    max_radius = 0.0
    
    for idx, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[idx] = 0.0
            continue
            
        # Normal vector to the facet
        n = cross / norm_cross
        
        # Ensure normal points outward (away from origin)
        if np.dot(n, a) < 0:
            n = -n
            
        d = np.dot(n, a)
        d = np.clip(d, -1.0, 1.0)
        r = np.arccos(d)
        radii[idx] = r
        
        if r > max_radius:
            max_radius = r
            
    return hull, radii, max_radius

def _smooth_objective_and_grad(x_flat, beta):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    """
    pts = x_flat.reshape(N, 3)
    pts = _to_sphere(pts)
    
    hull, radii, max_r = _compute_hull_and_radii(pts)
    if hull is None:
        return 10.0, np.zeros_like(x_flat)
    
    if len(radii) == 0:
        return 10.0, np.zeros_like(x_flat)
        
    r_max = np.max(radii)
    if r_max == 0:
        return 0.0, np.zeros_like(x_flat)
        
    exp_terms = np.exp(beta * (radii - r_max))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return r_max, np.zeros_like(x_flat)
        
    obj_val = r_max + (1.0 / beta) * np.log(sum_exp)
    
    grad_pts = np.zeros_like(pts)
    
    for idx, simplex in enumerate(hull.simplices):
        w = exp_terms[idx] / sum_exp
        if w < 1e-9:
            continue
            
        i, j, k = simplex
        a = pts[i]
        b = pts[j]
        c = pts[k]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            continue
            
        n = cross / norm_cross
        if np.dot(n, a) < 0:
            n = -n
            
        d = np.clip(np.dot(n, a), -1.0, 1.0)
        sin_r = np.sqrt(1.0 - d*d)
        if sin_r < 1e-6:
            sin_r = 1e-6
            
        g_vec = -n / sin_r
        
        grad_pts[i] += w * g_vec
        grad_pts[j] += w * g_vec
        grad_pts[k] += w * g_vec
        
    grad_x = np.zeros_like(x_flat)
    for i in range(N):
        p = pts[i]
        gp = grad_pts[i]
        grad_tangent = gp - np.dot(gp, p) * p
        grad_x[i*3:(i+1)*3] = grad_tangent
        
    return obj_val, grad_x

def _repulsion_energy_and_grad(x_flat, k=3.0):
    """
    Compute repulsion energy and gradient to encourage point spread.
    Energy: sum_{i != j} (1 - dot(x_i, x_j))^{-k}
    """
    pts = x_flat.reshape(N, 3)
    pts = _to_sphere(pts)
    
    D = np.dot(pts, pts.T)
    eps = 1e-8
    D_clipped = np.clip(D, -1.0, 1.0 - eps)
    
    mask = ~np.eye(N, dtype=bool)
    D_work = D_clipped[mask]
    
    terms = (1.0 - D_work) ** (-k)
    energy = np.sum(terms)
    
    # Gradient: d/dx_i sum_{j != i} (1 - x_i.x_j)^{-k} = k * sum_{j != i} (1 - x_i.x_j)^{-k-1} * x_j
    grad_pts = np.zeros_like(pts)
    inv_terms = (1.0 - D_work) ** (-(k + 1.0))
    
    # Reconstruct full matrix of inverse terms
    inv_full = np.zeros((N, N))
    inv_full[mask] = inv_terms
    
    for i in range(N):
        grad_pts[i] = k * np.sum(inv_full[i, :] * pts, axis=1)
    
    # Project to tangent space
    grad_x = np.zeros_like(x_flat)
    for i in range(N):
        p = pts[i]
        gp = grad_pts[i]
        grad_tangent = gp - np.dot(gp, p) * p
        grad_x[i*3:(i+1)*3] = grad_tangent
    
    return energy, grad_x

def _spherical_lloyd_iteration(pts, n_iters=50):
    """
    Spherical Lloyd relaxation: move each point to the centroid of its Voronoi cell
    on the sphere.
    """
    pts = _to_sphere(pts.copy())
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 30:
            break
            
        # Compute convex hull to identify facets
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # For each point, compute the weighted centroid of its adjacent facets
        new_pts = np.zeros_like(pts)
        
        # Build adjacency: which facets are adjacent to which points
        point_facets = [[] for _ in range(N)]
        for idx, simplex in enumerate(hull.simplices):
            for p_idx in simplex:
                point_facets[p_idx].append(idx)
        
        for i in range(N):
            if not point_facets[i]:
                new_pts[i] = pts[i]
                continue
                
            # Collect facet normals and areas
            weights = []
            normals = []
            for f_idx in point_facets[i]:
                simplex = hull.simplices[f_idx]
                a = pts[simplex[0]]
                b = pts[simplex[1]]
                c = pts[simplex[2]]
                
                v1 = a - b
                v2 = a - c
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                if norm_cross < 1e-12:
                    continue
                    
                n = cross / norm_cross
                if np.dot(n, a) < 0:
                    n = -n
                    
                # Area of triangle on sphere ~ norm_cross
                weights.append(norm_cross)
                normals.append(n)
                
            if not normals:
                new_pts[i] = pts[i]
                continue
                
            normals = np.array(normals)
            weights = np.array(weights)
            
            # Weighted centroid
            centroid = np.sum(normals * weights[:, np.newaxis], axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
        
        pts = _to_sphere(new_pts)
        
    return pts

def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _to_sphere(pts)

def _random_points(seed):
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _to_sphere(pts)

def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with jitter."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = _to_sphere(verts)
    
    # Create 50 points by subdividing and adding jitter
    np.random.seed(12345)
    pts = []
    for v in verts:
        # Add 4 points around each vertex with jitter
        for _ in range(4):
            jitter = np.random.randn(3) * 0.3
            pts.append(v + jitter)
    # Add 2 more points
    for _ in range(2):
        jitter = np.random.randn(3)
        pts.append(jitter)
    
    pts = np.array(pts[:N])
    return _to_sphere(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    best_pts = None
    best_obj = np.pi
    
    # Generate diverse initializations
    initializations = []
    initializations.append(_fibonacci_points())
    initializations.append(_icosahedron_based_points())
    for seed in [42, 123, 999, 555, 777, 2024]:
        initializations.append(_random_points(seed))
        
    for init_pts in initializations:
        if time.time() > _DEADLINE - 40:
            break
            
        # Phase 1: Repulsion refinement to spread points
        x0 = init_pts.flatten()
        try:
            result_rep = minimize(
                lambda x: _repulsion_energy_and_grad(x, k=3.0),
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            rep_pts = _to_sphere(result_rep.x.reshape(N, 3))
        except Exception:
            rep_pts = init_pts
        
        # Phase 2: Spherical Lloyd relaxation
        lloyd_pts = _spherical_lloyd_iteration(rep_pts, n_iters=30)
        
        # Phase 3: Multi-stage beta cascade optimization
        current_pts = lloyd_pts
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > _DEADLINE - 20:
                    break
                    
            x0 = current_pts.flatten()
            try:
                def obj_grad(x_flat, _beta=beta):
                    return _smooth_objective_and_grad(x_flat, _beta)
                    
                result = minimize(
                    obj_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1500}
                )
                
                if result.success:
                    opt_pts = _to_sphere(result.x.reshape(N, 3))
                    hull, radii, exact_obj = _compute_hull_and_radii(opt_pts)
                    if exact_obj < best_obj:
                        best_obj = exact_obj
                        best_pts = opt_pts.copy()
                        current_pts = opt_pts
            except Exception:
                pass
                
    # Final high-precision polish
    if best_pts is not None and time.time() < _DEADLINE - 10:
        x0 = best_pts.flatten()
        
        def _final_obj(x_flat):
            return _smooth_objective_and_grad(x_flat, beta=1000.0)
            
        try:
            result = minimize(
                _final_obj,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if result.fun < best_obj:
                opt_pts = _to_sphere(result.x.reshape(N, 3))
                hull, radii, exact_obj = _compute_hull_and_radii(opt_pts)
                if exact_obj < best_obj:
                    best_obj = exact_obj
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_points()
        
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
