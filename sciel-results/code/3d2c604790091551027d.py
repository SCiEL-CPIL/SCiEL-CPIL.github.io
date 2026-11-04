"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=120):
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

def _fast_facets(pts):
    """
    Compute facet data (circumcenters, circumradii, simplex indices) efficiently.
    Returns (ccs, crs, simplices) where ccs has shape (n_facets, 3),
    crs has shape (n_facets,), simplices has shape (n_facets, 3).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), dtype=int)
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # v = cross(a-b, b-c) gives circumcenter direction
    v = np.cross(a - b, b - c)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    valid = norm_v.squeeze() > 1e-12
    norm_v[~valid] = 1.0
    u = v / norm_v
    
    # n_face = cross(b-a, c-a) for orientation
    n_face = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n_face, axis=1, keepdims=True)
    valid2 = norm_n.squeeze() > 1e-12
    norm_n[~valid2] = 1.0
    n_face = n_face / norm_n
    
    # Orient normals outward
    dot_na = np.sum(n_face * a, axis=1)
    flip = dot_na < 0
    n_face[flip] *= -1
    
    # Choose circumcenter direction (should be on same side as outward normal)
    dot_un = np.sum(u * n_face, axis=1)
    cc = u.copy()
    cc[dot_un <= 0] *= -1
    
    # Compute circumradii
    dots_a = np.sum(cc * a, axis=1)
    dots_b = np.sum(cc * b, axis=1)
    dots_c = np.sum(cc * c, axis=1)
    dots_a = np.clip(dots_a, -1, 1)
    dots_b = np.clip(dots_b, -1, 1)
    dots_c = np.clip(dots_c, -1, 1)
    
    d_a = np.arccos(dots_a)
    d_b = np.arccos(dots_b)
    d_c = np.arccos(dots_c)
    crs = np.maximum(np.maximum(d_a, d_b), d_c)
    
    # Filter out invalid facets
    valid_all = valid & valid2
    if not np.all(valid_all):
        cc = cc[valid_all]
        crs = crs[valid_all]
        simplices = simplices[valid_all]
    
    return cc, crs, simplices

def _smooth_covering_radius_grad(pts, beta=50.0):
    """
    Compute soft-max covering radius and its gradient w.r.t. point positions.
    Uses the convex hull to find facets and their circumradii.
    Returns (soft_max_cr, gradient_array) where gradient has shape (n, 3).
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    cc, crs, simplices = _fast_facets(pts)
    
    if len(crs) == 0:
        return 1.0, np.zeros((n, 3))
    
    max_cr = np.max(crs)
    
    if max_cr < 1e-10:
        return max_cr, np.zeros((n, 3))
    
    # Soft-max: log(sum(exp(beta * crs))) / beta
    scaled = beta * (crs - max_cr)
    log_sum_exp = max_cr * beta + np.log(np.sum(np.exp(scaled)))
    soft_max_cr = log_sum_exp / beta
    
    # Weights for soft-max
    weights = np.exp(scaled) / np.sum(np.exp(scaled))
    
    # Compute gradient vectorized
    grad = np.zeros((n, 3))
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-15:
            continue
        cc_k = cc[k]
        for idx in simplices[k]:
            dot_val = np.dot(cc_k, pts[idx])
            dot_val = np.clip(dot_val, -1 + 1e-15, 1 - 1e-15)
            sin_d = np.sqrt(1 - dot_val * dot_val)
            if sin_d < 1e-15:
                continue
            vec = cc_k / sin_d
            vec_tan = vec - np.dot(vec, pts[idx]) * pts[idx]
            grad[idx] += w * vec_tan
    
    return soft_max_cr, grad

def _riesz_energy_grad(pts, s=2.0):
    """
    Compute Riesz s-energy and its gradient (vectorized).
    E = sum_{i != j} 1/|p_i - p_j|^s
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy
    inv_dist_s = 1.0 / np.power(dist, s)
    np.fill_diagonal(inv_dist_s, 0.0)
    energy = np.sum(inv_dist_s)
    
    # Gradient: dE/dp_i = s * sum_{j != i} (p_j - p_i) / |p_i - p_j|^(s+2)
    inv_dist_s2 = 1.0 / np.power(dist, s + 2)
    np.fill_diagonal(inv_dist_s2, 0.0)
    
    # Vectorized gradient computation
    # grad[i] = s * sum_j inv_dist_s2[i,j] * (pts[j] - pts[i])
    #         = s * (sum_j inv_dist_s2[i,j] * pts[j]) - s * (sum_j inv_dist_s2[i,j]) * pts[i]
    weight_sum = np.sum(inv_dist_s2, axis=1)  # shape (n,)
    weighted_pts = inv_dist_s2 @ pts  # shape (n, 3)
    grad = s * (weighted_pts - weight_sum[:, np.newaxis] * pts)
    
    # Project to tangent space
    grad_proj = grad - np.einsum('ij,ij->i', grad, pts)[:, np.newaxis] * pts
    
    return energy, grad_proj

def _spherical_lloyd(pts, max_iter=100, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
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
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
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

def _simulated_annealing(pts, start_temp=0.3, end_temp=0.01, n_iter=500, deadline=None):
    """
    Simulated annealing on the sphere. Accepts moves that increase covering radius
    with probability exp(-delta/T) to escape local minima.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    rng = np.random.RandomState(12345)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Pick a random point to move
        i = rng.randint(n)
        
        # Random perturbation: move in random tangent direction
        random_vec = rng.randn(3)
        tangent = random_vec - np.dot(random_vec, pts[i]) * pts[i]
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-10:
            continue
        tangent = tangent / norm_t
        
        # Temperature-dependent step size
        temp = start_temp * (end_temp / start_temp) ** (it / max(n_iter, 1))
        step = temp * 0.1
        
        new_pts = pts.copy()
        new_pts[i] = pts[i] + step * tangent
        new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        
        new_cr = _covering_radius(new_pts)
        delta = new_cr - current_cr
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_cr = new_cr
            if current_cr < best_cr:
                best_cr = current_cr
                best_pts = pts.copy()
    
    return _normalize(best_pts)

def _lbfgs_smooth(pts, deadline, beta_stages=None):
    """
    Multi-stage L-BFGS-B optimization using the smooth soft-max covering radius.
    """
    if beta_stages is None:
        beta_stages = [8, 20, 60, 200, 600, 1500]
    
    pts = _normalize(pts.copy())
    n = len(pts)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    bounds = [(-1.5, 1.5)] * (n * 3)
    
    for beta in beta_stages:
        if time.time() > deadline - 5:
            break
        
        def objective_and_grad(x, b=beta):
            p = x.reshape(n, 3)
            p = _normalize(p)
            val, grad = _smooth_covering_radius_grad(p, beta=b)
            # Project gradient to tangent space
            grad_proj = grad - np.einsum('ij,ij->i', grad, p)[:, np.newaxis] * p
            return val, grad_proj.ravel()
        
        x0 = best_pts.ravel()
        
        # Tighter tolerances for later stages, more iterations for high beta
        if beta <= 20:
            maxiter = 100
            ftol = 1e-13
            gtol = 1e-9
        elif beta <= 60:
            maxiter = 120
            ftol = 1e-14
            gtol = 1e-11
        elif beta <= 200:
            maxiter = 150
            ftol = 1e-15
            gtol = 1e-13
        else:
            maxiter = 200
            ftol = 1e-16
            gtol = 1e-15
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': maxiter,
                    'ftol': ftol,
                    'gtol': gtol,
                    'maxls': 80
                }
            )
            
            new_pts = _normalize(result.x.reshape(n, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
                
        except Exception:
            pass
    
    return _normalize(best_pts)

def _lbfgs_riesz(pts, deadline, s=2.0):
    """
    L-BFGS-B optimization using Riesz energy as a smooth auxiliary objective.
    This helps spread points more uniformly before the covering radius polish.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    bounds = [(-1.5, 1.5)] * (n * 3)
    
    def objective_and_grad(x):
        p = x.reshape(n, 3)
        p = _normalize(p)
        energy, grad = _riesz_energy_grad(p, s=s)
        # Project gradient to tangent space
        grad_proj = grad - np.einsum('ij,ij->i', grad, p)[:, np.newaxis] * p
        return energy, grad_proj.ravel()
    
    x0 = pts.ravel()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxls': 50
            }
        )
        
        new_pts = _normalize(result.x.reshape(n, 3))
        return new_pts
        
    except Exception:
        return pts

def _final_polish(pts, deadline):
    """
    High-precision final polish using very high beta soft-max and small steps.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    if time.time() > deadline - 3:
        return best_pts
    
    # Use very high beta for near-exact max, with two-stage approach
    for beta in [500.0, 2000.0]:
        if time.time() > deadline - 2:
            break
        
        def objective_and_grad(x, b=beta):
            p = x.reshape(n, 3)
            p = _normalize(p)
            val, grad = _smooth_covering_radius_grad(p, beta=b)
            grad_proj = grad - np.einsum('ij,ij->i', grad, p)[:, np.newaxis] * p
            return val, grad_proj.ravel()
        
        bounds = [(-1.5, 1.5)] * (n * 3)
        x0 = best_pts.ravel()
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-18,
                    'gtol': 1e-17,
                    'maxls': 100
                }
            )
            
            new_pts = _normalize(result.x.reshape(n, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
                
        except Exception:
            pass
    
    # Final tiny-step refinement with adaptive step size
    if time.time() < deadline - 1:
        beta = 3000.0
        step_size = 0.0005
        for it in range(80):
            if time.time() > deadline - 0.3:
                break
            
            val, grad = _smooth_covering_radius_grad(best_pts, beta=beta)
            grad_proj = grad - np.einsum('ij,ij->i', grad, best_pts)[:, np.newaxis] * best_pts
            
            grad_norms = np.linalg.norm(grad_proj, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-15] = 1.0
            
            improved = False
            for ls_factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
                cand = best_pts + step_size * ls_factor * (grad_proj / grad_norms)
                cand = _normalize(cand)
                cand_cr = _covering_radius(cand)
                if cand_cr < best_cr:
                    best_pts = cand
                    best_cr = cand_cr
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.4
                if step_size < 1e-12:
                    break
            else:
                step_size *= 0.998
    
    return _normalize(best_pts)

def _jittered_fibonacci_points(n, seed=42):
    """Fibonacci points with small random jitter for diversity."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_points(n)
    noise = 0.1 * rng.randn(n, 3)
    return _normalize(pts + noise)

def _random_points(n, seed=42):
    """Random uniform points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    # Define diverse initial configurations
    def _get_initial_config(restart_count):
        if restart_count == 1:
            return _fibonacci_points(N)
        elif restart_count == 2:
            return _jittered_fibonacci_points(N, seed=100 + restart_count)
        elif restart_count == 3:
            pts = _fibonacci_points(N)
            noise = 0.25 * np.random.randn(N, 3)
            return _normalize(pts + noise)
        elif restart_count == 4:
            return _random_points(N, seed=200 + restart_count)
        elif restart_count == 5:
            pts = _fibonacci_points(N)
            # Swap some points to break symmetry
            idx = np.random.permutation(N)[:6]
            pts[idx] = pts[idx[::-1]]
            noise = 0.12 * np.random.randn(N, 3)
            return _normalize(pts + noise)
        elif restart_count == 6:
            # Golden ratio based with different phase
            i = np.arange(N) + 0.3
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5
            r = np.sqrt(1.0 - z * z)
            return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart_count == 7:
            # Another variant with shifted z
            i = np.arange(N) + 0.7
            z = 1.0 - 2.0 * i / N
            z = np.clip(z, -1, 1)
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 1.0
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Perturb best solution
            noise_scale = 0.08 * (best_cr / 0.32)
            noise_scale = max(0.02, min(0.2, noise_scale))
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            return _normalize(pts)
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    restart_count = 0
    while time.time() < deadline - 35:
        restart_count += 1
        remaining = deadline - time.time()
        
        pts = _get_initial_config(restart_count)
        
        # Stage 1: Riesz energy minimization for uniform spreading (skip for later restarts)
        if remaining > 60 and restart_count <= 3:
            pts = _lbfgs_riesz(pts, deadline=deadline - 30, s=2.0)
        
        # Stage 2: Lloyd relaxation
        if time.time() < deadline - 30:
            lloyd_iters = min(80, max(30, int((deadline - time.time()) / 4)))
            pts = _spherical_lloyd(pts, max_iter=lloyd_iters)
        
        # Stage 3: Simulated annealing for basin jumping
        if time.time() < deadline - 25:
            sa_iters = min(300, int((deadline - time.time()) / 3))
            pts = _simulated_annealing(pts, start_temp=0.15, end_temp=0.003,
                                       n_iter=sa_iters, deadline=deadline - 8)
        
        # Stage 4: Multi-stage L-BFGS on smooth covering radius
        if time.time() < deadline - 20:
            pts = _lbfgs_smooth(pts, deadline=deadline - 8)
        
        # Stage 5: Final high-precision polish
        if time.time() < deadline - 12:
            pts = _final_polish(pts, deadline=deadline - 3)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
            # If significantly better, do extra polish
            if best_cr < 0.318 and time.time() < deadline - 20:
                best_pts = _final_polish(best_pts, deadline=deadline - 5)
                best_cr = _covering_radius(best_pts)
    
    # Final polish on best solution if time allows
    if time.time() < deadline - 10:
        best_pts = _final_polish(best_pts, deadline=deadline - 2)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
