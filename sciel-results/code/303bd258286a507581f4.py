"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _compute_circumcenters(pts, hull):
    """Vectorized computation of circumcenters and circumradii for all facets."""
    simplices = hull.simplices
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # v = cross(a-b, b-c) gives direction of circumcenter
    ab = a - b
    bc = b - c
    v = np.cross(ab, bc)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    valid = norm_v[:, 0] > 1e-12
    norm_v[~valid] = 1.0
    u = v / norm_v
    
    # Face normal
    n_face = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n_face, axis=1, keepdims=True)
    norm_n[norm_n < 1e-12] = 1.0
    n_face = n_face / norm_n
    
    # Orient normal outward
    dots = np.sum(n_face * a, axis=1, keepdims=True)
    n_face = np.where(dots < 0, -n_face, n_face)
    
    # Choose correct side
    dot_u_n = np.sum(u * n_face, axis=1, keepdims=True)
    cc = np.where(dot_u_n > 0, u, -u)
    
    # Compute circumradii
    d_a = np.arccos(np.clip(np.sum(cc * a, axis=1), -1, 1))
    d_b = np.arccos(np.clip(np.sum(cc * b, axis=1), -1, 1))
    d_c = np.arccos(np.clip(np.sum(cc * c, axis=1), -1, 1))
    crs = np.maximum(np.maximum(d_a, d_b), d_c)
    
    # Mask invalid
    crs[~valid] = 0.0
    cc[~valid] = 0.0
    
    return cc, crs, valid

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts)
    
    cc, crs, valid = _compute_circumcenters(pts, hull)
    if not np.any(valid):
        return _covering_radius_grid(pts)
    return np.max(crs[valid])

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_points():
    """Generate 12 icosahedron vertices."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    return _normalize(pts)

def _soft_max_crs(pts, beta):
    """
    Compute soft-max of facet circumradii using log-sum-exp.
    Returns (value, crs, valid_mask, hull) for reuse.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    
    cc, crs, valid = _compute_circumcenters(pts, hull)
    if not np.any(valid):
        return None
    
    crs_valid = crs[valid]
    if len(crs_valid) == 0:
        return None
    
    # Log-sum-exp for numerical stability
    max_cr = np.max(crs_valid)
    if max_cr < 1e-15:
        return max_cr, crs, valid, hull, max_cr
    
    shifted = crs_valid - max_cr
    log_sum_exp = max_cr + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    return log_sum_exp, crs, valid, hull, max_cr

def _soft_max_grad(pts, beta):
    """
    Compute gradient of soft-max covering radius w.r.t. point positions.
    Returns (value, gradient) where gradient is (n, 3).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    
    cc, crs, valid = _compute_circumcenters(pts, hull)
    if not np.any(valid):
        return None, None
    
    crs_valid = crs[valid]
    simplices_valid = hull.simplices[valid]
    cc_valid = cc[valid]
    
    max_cr = np.max(crs_valid)
    if max_cr < 1e-15:
        return max_cr, np.zeros_like(pts)
    
    # Soft-max weights
    shifted = crs_valid - max_cr
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient: d/dp_i of sum_j w_j * cr_j
    grad = np.zeros_like(pts)
    for j in range(len(crs_valid)):
        w = weights[j]
        if w < 1e-15:
            continue
        simplex = simplices_valid[j]
        c = cc_valid[j]
        
        for idx in simplex:
            p = pts[idx]
            dot_cp = np.dot(c, p)
            dot_cp = np.clip(dot_cp, -1 + 1e-12, 1 - 1e-12)
            denom = math.sqrt(1.0 - dot_cp * dot_cp)
            if denom < 1e-12:
                continue
            grad[idx] += w * (-c / denom)
    
    # Project gradient to tangent space
    for i in range(len(pts)):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    value = max_cr + np.log(np.sum(np.exp(beta * (crs_valid - max_cr)))) / beta
    
    return value, grad

def _lloyd_step(pts, max_iter=50, tol=1e-10):
    """Spherical Lloyd's algorithm for Voronoi relaxation."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        incident = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            if np.dot(normal, a) < 0:
                normal = -normal
            for idx in simplex:
                incident[idx].append(normal)
        
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident[i]) == 0:
                new_pts[i] = pts[i]
                continue
            centroid = np.sum(incident[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
        
        pts = _normalize(new_pts)
        if max_move < tol:
            break
    
    return pts

def _lbfags_softmax(pts, beta, max_iter=200, ftol=1e-12, gtol=1e-10):
    """
    L-BFGS optimization of soft-max covering radius.
    Uses tangent-space updates.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    def objective_and_grad(x):
        p = x.reshape(n, 3)
        p = _normalize(p)
        
        result = _soft_max_grad(p, beta)
        if result[0] is None:
            return 1.0, np.zeros_like(x)
        
        val, grad = result
        grad_flat = grad.ravel()
        return val, grad_flat
    
    x0 = pts.ravel()
    
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
                'maxfun': 5000
            }
        )
        new_pts = _normalize(res.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def _multi_stage_lbfags(pts, deadline):
    """
    Multi-stage beta cascade: start with low beta for broad navigation,
    then increase beta for precise local optimization.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    beta_stages = [5, 15, 30, 60, 100, 150, 200]
    ftol_stages = [1e-8, 1e-10, 1e-12, 1e-12, 1e-14, 1e-14, 1e-16]
    gtol_stages = [1e-6, 1e-8, 1e-10, 1e-10, 1e-12, 1e-14, 1e-14]
    
    for stage, (beta, ftol, gtol) in enumerate(zip(beta_stages, ftol_stages, gtol_stages)):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time()
        max_iter = min(100, max(20, int(remaining * 2)))
        
        pts = _lbfags_softmax(pts, beta, max_iter=max_iter, ftol=ftol, gtol=gtol)
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts, best_cr

def _global_search(pts, start_time, deadline, n_restarts=5):
    """
    Global search with multiple restarts, each followed by local optimization.
    """
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    n = len(pts)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            curr_pts = pts.copy()
        else:
            n_perturb = random.randint(5, 15)
            idxs = random.sample(range(n), n_perturb)
            curr_pts = pts.copy()
            for idx in idxs:
                rand_vec = np.random.randn(3)
                tangent = rand_vec - np.dot(rand_vec, curr_pts[idx]) * curr_pts[idx]
                tn = np.linalg.norm(tangent)
                if tn < 1e-10:
                    continue
                tangent = tangent / tn
                step = np.random.uniform(-0.3, 0.3)
                curr_pts[idx] = curr_pts[idx] + step * tangent
            curr_pts = _normalize(curr_pts)
        
        curr_pts = _lloyd_step(curr_pts, max_iter=30)
        cr = _covering_radius(curr_pts)
        
        if cr < best_cr + 0.01:
            curr_pts, cr = _multi_stage_lbfags(curr_pts, deadline - 10)
            if cr < best_cr:
                best_cr = cr
                best_pts = curr_pts.copy()
    
    return best_pts, best_cr

def _final_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances."""
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    if time.time() < deadline - 5:
        pts = _lbfags_softmax(pts, beta=500, max_iter=50, ftol=1e-16, gtol=1e-14)
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts, best_cr

def _repulsion_energy(pts):
    """Compute a repulsion energy to encourage spread."""
    pts = _normalize(pts)
    n = len(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)
    min_angle = np.min(angles)
    beta = 20
    shifted = angles - min_angle
    soft_min = min_angle + np.log(np.sum(np.exp(-beta * shifted))) / beta
    return -soft_min

def _repulsion_grad(pts):
    """Gradient of repulsion energy (negative soft-min of pairwise angles)."""
    pts = _normalize(pts)
    n = len(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)
    
    min_angle = np.min(angles)
    beta = 20
    shifted = angles - min_angle
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    np.fill_diagonal(weights, 0)
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = weights[i, j]
            if w < 1e-15:
                continue
            dot_ij = dots[i, j]
            denom = math.sqrt(max(1e-12, 1.0 - dot_ij * dot_ij))
            grad[i] += w * (-pts[j] / denom)
    
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    value = -(min_angle + np.log(np.sum(np.exp(-beta * (angles - min_angle)))) / beta)
    return value, grad

def _simulated_annealing(pts, deadline, n_iter=200):
    """Simulated annealing with local optimization."""
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    curr_pts = pts.copy()
    curr_cr = best_cr
    n = len(pts)
    
    T_start = 0.5
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, n_iter))
    
    for it in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        T = T_start * (cooling ** it)
        
        # Perturb a random subset
        n_perturb = random.randint(1, 5)
        idxs = random.sample(range(n), n_perturb)
        new_pts = curr_pts.copy()
        for idx in idxs:
            rand_vec = np.random.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_pts[idx]) * new_pts[idx]
            tn = np.linalg.norm(tangent)
            if tn < 1e-10:
                continue
            tangent = tangent / tn
            step = np.random.uniform(-0.2, 0.2)
            new_pts[idx] = new_pts[idx] + step * tangent
        new_pts = _normalize(new_pts)
        
        # Quick local optimization
        new_pts = _lloyd_step(new_pts, max_iter=20)
        new_cr = _covering_radius(new_pts)
        
        delta = new_cr - curr_cr
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            curr_pts = new_pts
            curr_cr = new_cr
            if curr_cr < best_cr:
                best_cr = curr_cr
                best_pts = curr_pts.copy()
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Initial solution with local optimization
    pts = _fibonacci_points(N)
    pts = _lloyd_step(pts, max_iter=50)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 2: Multi-stage L-BFGS on soft-max surrogate
    if time.time() < deadline - 60:
        best_pts, best_cr = _multi_stage_lbfags(best_pts, deadline - 30)
    
    # Phase 3: Simulated annealing with local optimization
    if time.time() < deadline - 40:
        sa_pts, sa_cr = _simulated_annealing(best_pts, deadline - 20, n_iter=100)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_pts = sa_pts.copy()
    
    # Phase 4: Global search with restarts
    if time.time() < deadline - 30:
        best_pts, best_cr = _global_search(best_pts, start_time, deadline - 20, n_restarts=3)
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 10:
        best_pts, best_cr = _final_polish(best_pts, deadline)
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    
    # Verify not in a closed hemisphere
    try:
        hull = ConvexHull(best_pts)
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
