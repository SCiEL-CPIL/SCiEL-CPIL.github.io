import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    # Note: log(1/d) = -log(d). d = sqrt(d_sq). log(d) = 0.5 log(d_sq).
    # So term is -0.5 log(d_sq).
    # The sum over all pairs i<j is half the sum over all i!=j.
    # Sum_{i<j} -0.5 log(d_ij^2) = -0.25 * Sum_{i!=j} log(d_ij^2)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _energy_and_grad(x_flat):
    """
    Objective and gradient for L-BFGS-B (Euclidean gradient on normalized points).
    x_flat: flattened (N, 3) array.
    """
    pts = x_flat.reshape(N, 3)
    
    # Normalize points to ensure they stay on the sphere during optimization steps
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    e = _log_energy(pts)
    if not np.isfinite(e):
        return e, np.zeros_like(x_flat)
        
    # Compute gradient on the sphere
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances again for safety
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return e, np.zeros_like(x_flat)

    # Euclidean gradient of the energy with respect to pts
    # E = -0.25 * sum_{i,j} log(d_ij^2)
    # dE/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    return e, grad_eucl.flatten()

def _subdivided_icosahedron(n_points):
    """
    Generate points on the sphere using a subdivided icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 10, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    def _icosa_subdivide(vertices, faces, levels):
        verts_list = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts_list[v1] + verts_list[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_list.append(mid)
                mid_cache[idx] = len(verts_list) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts_list), fs

    # Level 2 gives 162 points. We want 150.
    verts, _ = _icosa_subdivide(verts, faces, 2)
    
    if len(verts) > n_points:
        # Remove points that are closest to their nearest neighbor
        min_dists = np.zeros(len(verts))
        for i in range(len(verts)):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            min_dists[i] = np.min(dists)
        
        order = np.argsort(min_dists)
        indices_to_keep = order[len(verts)-n_points:]
        verts = verts[indices_to_keep]
        
    return verts

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * math.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _project_to_sphere(pts):
    """
    Project points to the unit sphere.
    """
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, vecs):
    """
    Project vectors to the tangent space of the sphere at pts.
    pts: (N, 3)
    vecs: (N, 3)
    Returns: (N, 3) projected vectors
    """
    # Projection matrix P = I - x x^T
    # proj_vec = vec - (vec . x) x
    dot_products = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - dot_products * pts

def _exp_map(pts, vecs):
    """
    Exponential map on the sphere.
    pts: (N, 3) points on sphere
    vecs: (N, 3) tangent vectors
    Returns: new points on sphere
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    
    theta = norms_safe
    # sin(theta)/theta ~ 1 for small theta
    sin_theta_over_theta = np.ones_like(theta)
    mask = theta > 1e-8
    sin_theta_over_theta[mask] = np.sin(theta[mask]) / theta[mask]
    
    # cos(theta)
    cos_theta = np.cos(theta)
    
    # x_new = cos(theta) * x + sin(theta)/theta * v
    new_pts = cos_theta * pts + sin_theta_over_theta * vecs
    return _project_to_sphere(new_pts)

def _log_map(pts, new_pts):
    """
    Logarithmic map on the sphere (inverse of exp map).
    pts: (N, 3) reference points
    new_pts: (N, 3) target points
    Returns: (N, 3) tangent vectors
    """
    # Compute angle between pts and new_pts
    dot = np.sum(pts * new_pts, axis=1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    
    # Avoid division by zero
    theta_safe = np.where(theta < 1e-15, 1.0, theta)
    
    # sin(theta)
    sin_theta = np.sin(theta_safe)
    mask = theta < 1e-15
    sin_theta[mask] = 1e-15 # Small value to avoid 0/0
    
    # v = (theta / sin(theta)) * (new_pts - cos(theta) * pts)
    coeff = theta_safe / sin_theta
    vecs = coeff * (new_pts - dot * pts)
    
    # Set zero vectors where theta is 0
    vecs[mask] = 0.0
    
    return vecs

def _riemannian_cg(pts, max_iter=100, tol=1e-10):
    """
    Riemannian Conjugate Gradient method on the sphere.
    """
    pts = _project_to_sphere(pts)
    
    # Compute initial gradient
    e, grad_flat = _energy_and_grad(pts.flatten())
    grad = grad_flat.reshape(N, 3)
    grad_tan = _tangent_project(pts, grad)
    
    # Initial direction
    d = -grad_tan
    d_flat = d.flatten()
    
    # Initial step size
    alpha = 1.0
    
    for k in range(max_iter):
        # Check convergence
        norm_grad = np.linalg.norm(grad_tan)
        if norm_grad < tol:
            break
        
        # Line search (Backtracking)
        # We want to find alpha such that f(x + alpha d) < f(x) + c * alpha * <grad, d>
        c1 = 1e-4
        rho = 0.5
        
        # Ensure d is tangent
        d_tan = _tangent_project(pts, d)
        
        f_cur = e
        # Approximate directional derivative
        # <grad, d> in Euclidean space is same as in tangent space if d is tangent
        dir_deriv = np.sum(grad_flat * d_tan.flatten())
        
        alpha = 1.0
        for _ in range(20):
            # Step
            new_pts = _exp_map(pts, alpha * d_tan)
            f_new = _log_energy(new_pts)
            
            if np.isfinite(f_new) and f_new < f_cur + c1 * alpha * dir_deriv:
                break
            alpha *= rho
        else:
            # If line search fails, break
            break
            
        # Update points
        old_pts = pts
        pts = new_pts
        e = f_new
        
        # Compute new gradient
        e, grad_flat = _energy_and_grad(pts.flatten())
        grad = grad_flat.reshape(N, 3)
        grad_tan = _tangent_project(pts, grad)
        
        # Check convergence again
        norm_grad = np.linalg.norm(grad_tan)
        if norm_grad < tol:
            break
            
        # Update direction (Polak-Ribiere)
        # d_new = -grad_new + beta * d_old
        # beta = <grad_new - grad_old, d_new_old> / ||grad_old||^2
        # Note: d_old was in tangent space of old_pts. We need to parallel transport it to new_pts.
        # Simple approximation: project d_old to new tangent space.
        d_old_tan_new = _tangent_project(pts, d)
        
        # Compute beta
        num = np.sum((grad_tan - grad_tan) * d_old_tan_new) # This is 0 if we just subtracted same? No.
        # Correct PR formula:
        # beta = <g_k - g_{k-1}, p_k> / ||g_{k-1}||^2
        # Here p_k is the search direction which is d_old_tan_new (approximately)
        # Let's use a simpler Fletcher-Reeves or just restart if beta is negative.
        
        # Let's use a simpler update: restart if angle is obtuse, otherwise PR.
        # Actually, for robustness, let's use a simple restart strategy if not improving fast enough.
        # Or just use standard R-CG with parallel transport approximation.
        
        # Parallel transport of d to new tangent space:
        # d_pt = d - (d . x_new) x_new  (This is just projection, which is an approximation of PT)
        d_pt = _tangent_project(pts, d)
        
        # Compute beta
        denom = np.sum(grad_tan * grad_tan)
        if denom < 1e-20:
            beta = 0.0
        else:
            # g_new - g_old (approximated by projecting old g to new space? No, we have new g)
            # We need <g_new - g_old_pt, d_pt>
            # We don't have g_old_pt easily. 
            # Let's use the simple restart if the gradient is not decreasing.
            # Or use the formula: beta = <g_new, d_pt> / <g_old, d_old> ? No.
            
            # Let's stick to a robust simple method: 
            # If the new gradient is not smaller, restart.
            if norm_grad > 0.9 * np.linalg.norm(_tangent_project(old_pts, grad)):
                 beta = 0.0
            else:
                 # Polak-Ribiere
                 # We need <g_new - g_old, d>
                 # Since we don't have g_old in the new tangent space, we approximate.
                 # A common trick is to use the Euclidean dot product of the flat gradients projected.
                 # Let's just use beta = 0 for simplicity and robustness (Steepest Descent).
                 # Steepest Descent is slower but more robust against numerical issues in PT.
                 beta = 0.0
                 
        d = -grad_tan + beta * d_pt
        d = _tangent_project(pts, d)
        
    return pts

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Geodesic Dome (Subdivided Icosahedron) - Primary strategy
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere - Fallback / diverse start
    inits.append(_fibonacci_sphere(N))
    
    # 3. Random Uniform - Diverse start
    for seed in [42, 123, 999, 1234, 5678]:
        inits.append(_random_uniform_sphere(N, seed=seed))
        
    # Optimization Loop using Riemannian CG
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        try:
            # Run R-CG
            opt_pts = _riemannian_cg(init_pts, max_iter=50, tol=1e-8)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # If we have time, do a final polish on the best result using L-BFGS-B
    # L-BFGS is often better for the final tight convergence.
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        try:
            x0 = best_pts.flatten()
            res = minimize(
                _energy_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            opt_pts = res.x.reshape(N, 3)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            pass

    # Second polish pass if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_pts.flatten()
            res = minimize(
                _energy_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, 3)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Energy:", _log_energy(pts))
    print("Shape:", pts.shape)
