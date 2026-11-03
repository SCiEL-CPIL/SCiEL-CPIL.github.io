"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        return _covering_radius_fallback(pts)
    
    if len(simplices) == 0:
        return _covering_radius_fallback(pts)
        
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        return _covering_radius_fallback(pts)
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return np.degrees(np.max(radii))


def _covering_radius_fallback(pts):
    """Fallback: approximate covering radius by sampling the sphere."""
    n_samples = 5000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)
    min_angles = np.arccos(min_dots)
    return np.degrees(min_angles.max())


def _fibonacci_init():
    """Generate Fibonacci sphere initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _objective_and_grad(x, t=1.0):
    """
    Objective: minimize -min_eigenvalue(Gram matrix) + t * sum of squared norms deviation.
    Gradient computed analytically.
    """
    pts = x.reshape(N, 3)
    # Ensure unit norm by projection
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    # Gram matrix
    G = pts @ pts.T
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    min_eig = eigvals[0]
    min_vec = eigvecs[:, 0]
    
    # Gradient of min eigenvalue w.r.t. G: v v^T
    # dG/dx: dG_ij = p_i^T dp_j + p_j^T dp_i
    # d(min_eig)/dx_k = 2 * sum_{i,j} (v_i v_j) * (dG_ij/dx_k)
    # = 2 * (V V^T pts)_k where V is the eigenvector matrix
    # More precisely: d(min_eig)/d(pts) = 2 * (v v^T) @ pts
    grad_G = 2.0 * np.outer(min_vec, min_vec)
    grad_pts = grad_G @ pts
    
    # Objective: -min_eig
    obj = -min_eig
    
    # Gradient of objective
    grad = -grad_pts.flatten()
    
    return obj, grad


def _lloyd_step(points, n_iters=50):
    """
    Lloyd's algorithm variant: move each point toward the centroid of its Voronoi cell
    (approximated by the mean of neighboring points in the convex hull).
    """
    points = _normalize(points)
    
    for _ in range(n_iters):
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
        except Exception:
            break
        
        # For each point, find adjacent facets
        point_facets = [[] for _ in range(N)]
        for m, simplex in enumerate(simplices):
            for idx in simplex:
                point_facets[idx].append(m)
        
        # Compute facet circumcenters
        a = points[simplices[:, 0]]
        b = points[simplices[:, 1]]
        c = points[simplices[:, 2]]
        
        n = np.cross(b - a, c - a)
        norms = np.linalg.norm(n, axis=1)
        valid = norms > 1e-10
        if not np.all(valid):
            break
            
        u = n / norms[:, np.newaxis]
        centroids = (a + b + c) / 3.0
        dots = np.sum(u * centroids, axis=1)
        u[dots < 0] *= -1
        
        # For each point, move toward the average of its facet circumcenters
        new_points = points.copy()
        for i in range(N):
            if len(point_facets[i]) == 0:
                continue
            centers = u[point_facets[i]]
            avg_center = np.mean(centers, axis=0)
            avg_center = _normalize(avg_center.reshape(1, 3))[0]
            
            # Move toward avg_center in tangent space
            tangent = avg_center - np.dot(avg_center, points[i]) * points[i]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-12:
                step = 0.1 * tangent / norm_t
                new_points[i] += step
        
        new_points = _normalize(new_points)
        
        # Check for improvement
        old_radius = _covering_radius(points)
        new_radius = _covering_radius(new_points)
        
        if new_radius < old_radius:
            points = new_points
        else:
            break
    
    return points


def _nelder_mead_refine(points, time_budget=10.0):
    """
    Nelder-Mead refinement on the flattened point array, constrained to unit sphere.
    """
    start = time.time()
    x0 = points.flatten().copy()
    
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts)
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_pts = _normalize(result.x.reshape(N, 3))
    best_radius = _covering_radius(best_pts)
    
    if time.time() - start < time_budget:
        # Try a few more iterations with tighter tolerance
        result2 = minimize(objective, result.x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
        pts2 = _normalize(result2.x.reshape(N, 3))
        r2 = _covering_radius(pts2)
        if r2 < best_radius:
            best_pts = pts2
            best_radius = r2
    
    return best_pts


def _spectral_refinement(points, time_budget=15.0):
    """
    Refine using gradient descent on the spectral objective.
    """
    start = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    lr = 0.01
    for _ in range(200):
        if time.time() - start > time_budget:
            break
            
        obj, grad = _objective_and_grad(pts.flatten())
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        grad = grad / grad_norm
        
        # Update
        new_pts = _normalize((pts - lr * grad.reshape(N, 3)))
        
        new_radius = _covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            lr *= 0.99
        else:
            lr *= 0.8
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_radius = np.pi
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    init_fib = _fibonacci_init()
    initializations.append(init_fib)
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand = _normalize(init_rand)
    initializations.append(init_rand)
    
    # 3. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise = _normalize(init_fib_noise)
    initializations.append(init_fib_noise)
    
    # 4-8. More randomized restarts
    for s in range(5):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.05 * s)
        init = _normalize(init)
        initializations.append(init)
    
    # 9-13. Pure random with different seeds
    for s in range(5):
        np.random.seed(3000 + s * 23)
        init = np.random.randn(N, 3)
        init = _normalize(init)
        initializations.append(init)
    
    # Sort initializations by their initial covering radius (best first)
    init_radii = []
    for init in initializations:
        r = _covering_radius(init)
        init_radii.append((r, init))
    init_radii.sort(key=lambda x: x[0])
    
    # Local optimization on each initialization
    for radius, init_pts in init_radii:
        if time.time() > deadline - 15.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Try multiple optimization strategies
        candidates = []
        
        # Strategy 1: Lloyd's algorithm
        if time.time() < deadline - 10.0:
            lloyd_result = _lloyd_step(init_pts.copy(), n_iters=30)
            lloyd_radius = _covering_radius(lloyd_result)
            candidates.append((lloyd_radius, lloyd_result))
        
        # Strategy 2: Spectral refinement
        if time.time() < deadline - 10.0:
            spectral_budget = min(remaining * 0.2, 10.0)
            if spectral_budget > 2.0:
                spectral_result = _spectral_refinement(init_pts.copy(), time_budget=spectral_budget)
                spectral_radius = _covering_radius(spectral_result)
                candidates.append((spectral_radius, spectral_result))
        
        # Strategy 3: Nelder-Mead
        if time.time() < deadline - 10.0:
            nm_budget = min(remaining * 0.2, 8.0)
            if nm_budget > 2.0:
                nm_result = _nelder_mead_refine(init_pts.copy(), time_budget=nm_budget)
                nm_radius = _covering_radius(nm_result)
                candidates.append((nm_radius, nm_result))
        
        # Select the best candidate
        if candidates:
            candidates.sort(key=lambda x: x[0])
            cand_radius, cand_pts = candidates[0]
            
            if cand_radius < best_radius:
                best_radius = cand_radius
                best_pts = cand_pts.copy()
    
    # Final polish on the best result using Nelder-Mead
    if best_pts is not None and time.time() < deadline - 5.0:
        polish_budget = min(deadline - time.time() - 5.0, 10.0)
        if polish_budget > 2.0:
            polished = _nelder_mead_refine(best_pts.copy(), time_budget=polish_budget)
            polished_radius = _covering_radius(polished)
            
            if polished_radius < best_radius:
                best_radius = polished_radius
                best_pts = polished.copy()
    
    if best_pts is None:
        best_pts = init_radii[0][1]
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
