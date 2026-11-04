"""Sphere covering with 50 points - optimized algorithm with Gaussian quadrature."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize
from scipy.special import lpmv, roots_legendre

# Precompute Gaussian quadrature nodes on the sphere for smooth distance approximation
# We use a product of 1D Gauss-Legendre nodes in spherical coordinates
def _make_spherical_quadrature(n_theta=100, n_phi=200):
    """Create quadrature points on the sphere using Gauss-Legendre in polar angle
    and uniform in azimuth."""
    # Gauss-Legendre nodes and weights for cos(theta) in [-1, 1]
    nodes, weights = roots_legendre(n_theta)
    cos_theta = nodes
    sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta**2))
    theta = np.arccos(cos_theta)
    
    # Uniform azimuth
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    
    # Convert to Cartesian
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    # Build 3D points: (theta, phi) -> (x, y, z)
    # x = st * cp, y = st * sp, z = ct
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    st_grid = np.sin(theta_grid)
    ct_grid = np.cos(theta_grid)
    cp_grid = np.cos(phi_grid)
    sp_grid = np.sin(phi_grid)
    
    x = st_grid * cp_grid
    y = st_grid * sp_grid
    z = ct_grid
    
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Weights: product of 1D weights * (1/(2*pi)) for uniform phi
    w_theta = weights  # weights for cos(theta) integration
    w_phi = 1.0 / n_phi  # uniform azimuth
    
    weights_2d = np.outer(w_theta, w_phi)
    weights_flat = weights_2d.ravel()
    
    # Normalize weights to sum to 1
    weights_flat /= weights_flat.sum()
    
    return points, weights_flat

# Precompute quadrature
_QUAD_POINTS, _QUAD_WEIGHTS = _make_spherical_quadrature(n_theta=80, n_phi=160)

def _normalize(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r

def _smooth_distance_field(x, temperature=0.1):
    """Compute smooth approximation of max distance using Gaussian quadrature.
    Returns (smooth_max, gradient) where gradient is w.r.t. x flattened."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute angular distances from each quadrature point to nearest configuration point
    # d(i, j) = arccos(clip(q_i . p_j, -1, 1))
    # For efficiency, compute dot products
    dots = _QUAD_POINTS @ pts.T  # shape (n_quad, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # shape (n_quad, N)
    
    # For each quadrature point, distance to nearest config point
    min_dists = np.min(dists, axis=1)  # shape (n_quad,)
    
    # Smooth max via log-sum-exp with temperature
    T = temperature
    scaled = min_dists / T
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum_exp * T
    
    # Gradient: d(smooth_max)/d(p_k) = sum_i w_i * d(min_d_i)/d(p_k)
    # where w_i = exp(scaled_i - max_scaled) / sum(exp(scaled_j - max_scaled)) * (1/T)
    # d(min_d_i)/d(p_k) = -1/sin(d_ik) * (q_i - (q_i . p_k) p_k) / |...| ... 
    # Actually, d(arccos(q.p))/d(p) = -q / sin(arccos(q.p)) projected tangent
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    weights = weights / T  # chain rule factor
    
    grad_pts = np.zeros_like(pts)
    
    # For each quadrature point, find which config point is nearest
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dists = min_dists
    
    # Accumulate gradient
    for i in range(len(_QUAD_POINTS)):
        if weights[i] < 1e-15:
            continue
        q = _QUAD_POINTS[i]
        k = nearest_idx[i]
        d = nearest_dists[i]
        sin_d = np.sin(d)
        if sin_d < 1e-12:
            continue
        # d(arccos(q.p_k))/d(p_k) = -q/sin(d) (tangent component)
        grad_vec = -q / sin_d
        # Project to tangent plane at p_k
        grad_tangent = grad_vec - np.dot(grad_vec, pts[k]) * pts[k]
        grad_pts[k] += weights[i] * grad_tangent
    
    return smooth_max, grad_pts.flatten()

def _lbfgs_quadrature(points, time_limit=None, start_time=None, temperature=0.1, maxiter=500):
    """Refine points using L-BFGS-B with Gaussian quadrature smooth objective."""
    points = _normalize(points)
    
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _smooth_distance_field(x, temperature)[0],
            x0,
            jac=lambda x: _smooth_distance_field(x, temperature)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_points = new_pts
    except Exception:
        pass
    
    return best_points, best_cr

def _multi_temp_quadrature(points, time_limit=None, start_time=None, temps=None):
    """Multi-stage temperature cascade for quadrature-based refinement."""
    if temps is None:
        temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for T in temps:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        pts, cr = _lbfgs_quadrature(best_points, time_limit=time_limit, start_time=start_time, temperature=T, maxiter=300)
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    return best_points, best_cr

def _riesz_energy_grad(x, s=3.0):
    """Compute Riesz s-energy and its gradient."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    G = pts @ pts.T
    np.fill_diagonal(G, 0.0)
    
    dists_sq = 2.0 * (1.0 - G)
    dists_sq = np.maximum(dists_sq, 1e-15)
    dists = np.sqrt(dists_sq)
    
    energy = np.sum(dists ** (-s))
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = dists[i, j]
            grad[i] += -s * diff / (d ** (s + 2))
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad.flatten()

def _lbfgs_riesz(points, time_limit=None, start_time=None, s=3.0, maxiter=200):
    """Refine points using L-BFGS-B with Riesz energy objective."""
    points = _normalize(points)
    
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _riesz_energy_grad(x, s)[0],
            x0,
            jac=lambda x: _riesz_energy_grad(x, s)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_points = new_pts
    except Exception:
        pass
    
    return best_points, best_cr

def _lloyd_iteration(points, max_iters=2000, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = _normalize(points.copy())
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        centers = []
        radii = []
        simplices_list = []
        
        for simplex in hull.simplices:
            pts = points[simplex]
            if len(pts) != 3:
                continue
            a, b, c = pts[0], pts[1], pts[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(dot_a, -1.0, 1.0)
            r = np.arccos(cos_r)
            centers.append(v)
            radii.append(r)
            simplices_list.append(simplex)
        
        if not centers:
            break
        
        new_points = np.zeros_like(points)
        
        point_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(simplices_list):
            for idx in simplex:
                point_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = [centers[fi] for fi in point_facets[i]]
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

def _fibonacci_points(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(n, offset=0.5, jitter=0.1, seed=42):
    """Fibonacci points with random jitter."""
    pts = _fibonacci_points(n, offset)
    rng = np.random.RandomState(seed)
    for i in range(n):
        noise = rng.randn(3) * jitter
        pts[i] = pts[i] + noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts

def _icosahedron_based_points(n, seed=123):
    """Generate points based on icosahedron vertices expanded to n points."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    rng = np.random.RandomState(seed)
    points = []
    for i in range(n):
        random_dir = rng.randn(3)
        random_dir /= np.linalg.norm(random_dir)
        dots = ico @ random_dir
        closest = ico[np.argmax(dots)]
        
        noise = rng.randn(3) * 0.3
        p = closest + noise
        p /= np.linalg.norm(p)
        points.append(p)
    
    return np.array(points)

def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _perturb_and_refine(points, n_perturbs=10, time_limit=None, start_time=None):
    """Perturb random points and refine with local optimization."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(42)
    
    for _ in range(n_perturbs):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 6)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * 0.08
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        refined, cr1 = _lloyd_iteration(perturbed, max_iters=500, time_limit=5, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        refined, cr2 = _multi_temp_quadrature(refined, time_limit=8, start_time=start_time, temps=[0.5, 0.2, 0.1, 0.05])
        
        if cr2 < best_cr:
            best_cr = cr2
            best_points = refined.copy()
    
    return best_points, best_cr

def _final_polish(points, time_limit=None, start_time=None):
    """High-precision final polish using very low temperature quadrature and tight Lloyd."""
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    remaining = time_limit if time_limit else 10
    if start_time and time.time() - start_time < remaining - 3:
        pts, cr = _lloyd_iteration(best_points, max_iters=5000, tol=1e-13, time_limit=3, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    if start_time and time.time() - start_time < remaining - 2:
        pts, cr = _lbfgs_quadrature(best_points, time_limit=2, start_time=start_time, temperature=0.005, maxiter=500)
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    if start_time and time.time() - start_time < remaining - 1:
        pts, cr = _lloyd_iteration(best_points, max_iters=2000, tol=1e-14, time_limit=1, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    return best_points, best_cr

def _multi_start_refine(start_points_list, time_limit=None, start_time=None):
    """Try multiple starting points and keep the best result."""
    best_points = None
    best_cr = np.inf
    
    for sp in start_points_list:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        refined, cr1 = _lloyd_iteration(sp, max_iters=1000, time_limit=10, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        refined, cr_riesz = _lbfgs_riesz(refined, time_limit=8, start_time=start_time, s=3.0)
        
        refined, cr2 = _multi_temp_quadrature(refined, time_limit=15, start_time=start_time, temps=[1.0, 0.5, 0.2, 0.1, 0.05])
        
        if cr2 < best_cr:
            best_cr = cr2
            best_points = refined.copy()
    
    return best_points, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate diverse starting points
    starts = [
        _fibonacci_points(N, offset=0.5),
        _fibonacci_points(N, offset=0.0),
        _fibonacci_points(N, offset=1.0),
        _jittered_fibonacci_points(N, offset=0.5, jitter=0.15, seed=42),
        _jittered_fibonacci_points(N, offset=0.5, jitter=0.15, seed=123),
        _icosahedron_based_points(N, seed=123),
        _icosahedron_based_points(N, seed=456),
        _random_points(N, seed=42),
        _random_points(N, seed=123),
        _random_points(N, seed=789),
    ]
    
    # Phase 1: Multi-start with Lloyd + Riesz + quadrature-based refinement
    remaining = deadline - time.time()
    if remaining > 30:
        best_points, best_cr = _multi_start_refine(starts, time_limit=remaining * 0.45, start_time=start_time)
        if best_points is None:
            best_points = _fibonacci_points(N)
            best_cr = _covering_radius(best_points)
    else:
        best_points = _fibonacci_points(N)
        best_cr = _covering_radius(best_points)
    
    # Phase 2: Perturb and refine from best solution
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _perturb_and_refine(best_points, n_perturbs=20, time_limit=remaining * 0.4, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        points, cr = _final_polish(best_points, time_limit=remaining * 0.5, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
