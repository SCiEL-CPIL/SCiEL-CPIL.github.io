"""Improved program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

def _facet_circumradius_batch(pts, simplices):
    """Vectorized computation of facet circumradii for all simplices at once."""
    n_facets = len(simplices)
    if n_facets == 0:
        return np.array([])
    
    v1 = pts[simplices[:, 0]]
    v2 = pts[simplices[:, 1]]
    v3 = pts[simplices[:, 2]]
    
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1, e2)
    n_norms = np.linalg.norm(normals, axis=1)
    
    valid = n_norms > 1e-15
    angles = np.zeros(n_facets)
    
    if np.any(valid):
        d_vals = np.einsum('ij,ij->i', normals[valid], v1[valid])
        dists = np.abs(d_vals) / n_norms[valid]
        dists = np.clip(dists, -1.0, 1.0)
        angles[valid] = np.arccos(dists)
    
    return angles

def _facet_circumcenters(pts, simplices):
    """Vectorized computation of facet circumcenters (unit vectors)."""
    n_facets = len(simplices)
    if n_facets == 0:
        return np.zeros((0, 3))
    
    v1 = pts[simplices[:, 0]]
    v2 = pts[simplices[:, 1]]
    v3 = pts[simplices[:, 2]]
    
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1, e2)
    n_norms = np.linalg.norm(normals, axis=1)
    
    valid = n_norms > 1e-15
    centers = np.zeros((n_facets, 3))
    
    if np.any(valid):
        d_vals = np.einsum('ij,ij->i', normals[valid], v1[valid])
        centers_valid = (d_vals[:, None] / (n_norms[valid] ** 2)[:, None]) * normals[valid]
        c_norms = np.linalg.norm(centers_valid, axis=1)
        valid2 = c_norms > 1e-15
        centers_valid[valid2] /= c_norms[valid2, None]
        
        valid_indices = np.where(valid)[0]
        centers[valid_indices[valid2]] = centers_valid[valid2]
    
    return centers

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return _covering_radius_grid(pts, 360, 720)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts, 360, 720)
    
    angles = _facet_circumradius_batch(pts, hull.simplices)
    if len(angles) == 0:
        return 0.0
    return np.degrees(np.max(angles))

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_grid(pts, n_theta=180, n_phi=360):
    """Compute covering radius in degrees using a grid approximation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    max_radius_rad = np.arccos(np.min(min_dots))
    return np.degrees(max_radius_rad)

def _soft_covering_radius_and_grad(pts, beta=50.0, n_theta=90, n_phi=180):
    """
    Compute a smooth log-sum-exp surrogate for the covering radius and its gradient.
    The covering radius is max_y arccos(max_i y·p_i).
    We approximate with a grid and use log-sum-exp for smoothness.
    Returns (radius_degrees, gradient_array).
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])  # (M, 3)
    
    # dots: (M, N)
    dots = grid @ pts.T
    dots = np.clip(dots, -0.9999999, 0.9999999)
    
    # For each grid point, the nearest point gives the max dot product
    # We want to minimize max over grid of arccos(max_i dot)
    # = minimize max over grid of -log(max_i dot) approximately
    # Better: use log-sum-exp of exp(-beta * dot) to approximate max of -dot
    
    # Actually, let's use: f = (1/beta) * log(sum_i exp(-beta * dot_ij)) for each j
    # This approximates max_i (-dot_ij) = -min_i dot_ij... wait that's wrong.
    # We want max_i dot_ij, so: (1/beta)*log(sum_i exp(beta * dot_ij)) approximates max_i dot_ij
    
    # For each grid point j, g_j = (1/beta) * log(sum_i exp(beta * dot_ij)) approximates max_i dot_ij
    # Then covering radius approx = max_j arccos(g_j)
    # Use log-sum-exp again: (1/beta2) * log(sum_j exp(beta2 * arccos(g_j)))
    
    # Simpler approach: minimize sum of exp(beta * (arccos(max_dot_j) - t)) for some t
    # Or just: minimize (1/beta) * log(sum_j exp(beta * arccos(max_dot_j)))
    
    # Let's use: objective = (1/beta) * log(sum_j exp(beta * angle_j)) where angle_j = arccos(max_dot_j)
    # This is a smooth upper bound on max_j angle_j
    
    max_dots = np.max(dots, axis=1)  # (M,)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))  # (M,)
    
    # Log-sum-exp for stability
    max_angle = np.max(angles)
    if max_angle < 1e-15:
        return 0.0, np.zeros_like(pts)
    
    exp_angles = np.exp(beta * (angles - max_angle))
    sum_exp = np.sum(exp_angles)
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(pts)
    
    log_sum_exp = max_angle + np.log(sum_exp) / beta
    objective = np.degrees(log_sum_exp)
    
    # Gradient: d/dp_k [ (1/beta) log(sum_j exp(beta * angle_j)) ]
    # = (1/beta) * sum_j [ beta * exp(beta*angle_j)/sum_exp * d(angle_j)/dp_k ]
    # = sum_j [ w_j * d(angle_j)/dp_k ]
    # where w_j = exp(beta*angle_j)/sum_exp
    # angle_j = arccos(max_i dot_ij)
    # d(angle_j)/dp_k = -1/sqrt(1-dot_kj^2) * d(dot_kj)/dp_k if k is argmax
    # But max is not differentiable. Use soft version:
    # Replace max with log-sum-exp: g_j = (1/beta2) log(sum_i exp(beta2 * dot_ij))
    # Then angle_j = arccos(g_j)
    # d(g_j)/dp_k = (1/beta2) * (1/sum_i exp(beta2*dot_ij)) * beta2 * exp(beta2*dot_kj)
    #              = exp(beta2*dot_kj) / sum_i exp(beta2*dot_ij)
    
    beta2 = 20.0  # inner soft-max temperature
    exp_dots = np.exp(beta2 * (dots - dots.max(axis=1, keepdims=True)))
    soft_max_weights = exp_dots / exp_dots.sum(axis=1, keepdims=True)  # (M, N)
    
    # g_j = sum_i soft_max_weights[j,i] * dot[j,i]  (approximation to max)
    # Actually g_j = (1/beta2) * log(sum_i exp(beta2 * dot_ij))
    # For gradient, we need d(g_j)/dp_k = soft_max_weights[j,k]
    
    # angle_j = arccos(g_j)
    # d(angle_j)/dp_k = -soft_max_weights[j,k] / sqrt(1 - g_j^2)
    
    g_vals = np.sum(soft_max_weights * dots, axis=1)  # (M,)
    g_vals = np.clip(g_vals, -0.9999999, 0.9999999)
    sqrt_terms = np.sqrt(np.maximum(1.0 - g_vals**2, 1e-15))  # (M,)
    
    # w_j from outer log-sum-exp
    w_j = exp_angles / sum_exp  # (M,)
    
    # Gradient contribution: -sum_j w_j / sqrt(1-g_j^2) * soft_max_weights[j,k] * grid[j,:]
    # For each point k: grad[k] = -sum_j [w_j / sqrt(1-g_j^2)] * soft_max_weights[j,k] * grid[j,:]
    
    coeff = -w_j / sqrt_terms  # (M,)
    # grad[k] = -sum_j coeff[j] * soft_max_weights[j,k] * grid[j,:]
    # = -(coeff[:, None] * soft_max_weights) @ grid  -> (N, 3)
    
    grad = -(coeff[:, None] * soft_max_weights) @ grid
    grad = np.deg2rad(1.0) * grad  # convert from degrees gradient to radians
    
    # Normalize points to stay on sphere: project gradient
    # grad_projected[k] = grad[k] - (grad[k]·p_k) * p_k
    dots_p = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dots_p * pts
    
    return objective, grad

def _lloyd_iteration(pts, n_theta=360, n_phi=720):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def _refine_lloyd(pts, start_time, max_time=120.0, max_iters=30):
    """Lloyd-based refinement with annealing."""
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_grid(current, 180, 360)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        new_pts = _lloyd_iteration(current)
        
        # Check for coincident points and nudge apart
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(new_pts[i], new_pts[j])
                if dot > 0.999999:
                    diff = new_pts[i] - new_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    new_pts[i] = new_pts[i] + 0.01 * diff
                    new_pts[j] = new_pts[j] - 0.01 * diff
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
        
        new_radius = _covering_radius_grid(new_pts, 180, 360)
        if new_radius < best_radius:
            current = new_pts
            best = new_pts.copy()
            best_radius = new_radius
        else:
            # Try blending
            alpha = 0.5
            blended = (1 - alpha) * current + alpha * new_pts
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            blended = blended / norms
            blended_radius = _covering_radius_grid(blended, 180, 360)
            if blended_radius < best_radius:
                current = blended
                best = blended.copy()
                best_radius = blended_radius
            else:
                break
    
    return best, best_radius

def _gradient_descent_facets(pts, start_time, max_time=120.0, max_iters=500):
    """
    Gradient descent targeting the worst facets using vectorized computations.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    step_size = 0.02
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 15.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        simplices = hull.simplices
        angles = _facet_circumradius_batch(best_pts, simplices)
        
        if len(angles) == 0:
            break
        
        max_angle = np.max(angles)
        if max_angle < np.radians(11.3):
            break
        
        # Soft-max weights over facets
        beta = 30.0
        exp_angles = np.exp(beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Compute circumcenters for all facets
        centers = _facet_circumcenters(best_pts, simplices)
        
        # Vectorized gradient computation
        grad = np.zeros_like(best_pts)
        
        # For each facet, we want to add weight * (center - proj_center_on_vertex) / norm
        # We can compute the projection of centers onto vertices for all pairs in the simplices
        # simplices: (F, 3), centers: (F, 3), best_pts: (N, 3)
        
        # v1, v2, v3 for each facet
        v1 = best_pts[simplices[:, 0]]
        v2 = best_pts[simplices[:, 1]]
        v3 = best_pts[simplices[:, 2]]
        
        # dot products
        d1 = np.einsum('ij,ij->i', v1, centers)
        d2 = np.einsum('ij,ij->i', v2, centers)
        d3 = np.einsum('ij,ij->i', v3, centers)
        
        # directions
        dir1 = centers - d1[:, None] * v1
        dir2 = centers - d2[:, None] * v2
        dir3 = centers - d3[:, None] * v3
        
        # norms
        n1 = np.linalg.norm(dir1, axis=1)
        n2 = np.linalg.norm(dir2, axis=1)
        n3 = np.linalg.norm(dir3, axis=1)
        
        # safe divide
        n1[n1 < 1e-10] = 1.0
        n2[n2 < 1e-10] = 1.0
        n3[n3 < 1e-10] = 1.0
        
        # weighted directions
        w = weights
        wd1 = (w / n1)[:, None] * dir1
        wd2 = (w / n2)[:, None] * dir2
        wd3 = (w / n3)[:, None] * dir3
        
        # Scatter add to grad
        # np.add.at is slow, but for N=130 and F~260 it's fine.
        # Or use bincount-like logic? No, vectors are 3D.
        # We can use advanced indexing with accumulation?
        # grad[idx] += val
        # Since indices are small, we can just loop over facets?
        # F is ~260. Looping 260 times is fast.
        # But we can do better.
        
        # Let's use a simple loop for scattering, it's only 260 iterations
        for i in range(len(simplices)):
            idx1, idx2, idx3 = simplices[i]
            grad[idx1] += wd1[i]
            grad[idx2] += wd2[i]
            grad[idx3] += wd3[i]
        
        # Apply gradient step
        if np.linalg.norm(grad) > 1e-15:
            grad_norm = np.linalg.norm(grad)
            grad = grad / grad_norm * step_size
            
            new_pts = best_pts + grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-10:
                best_pts = new_pts
                best_radius = new_radius
                no_improve_count = 0
                step_size = min(step_size * 1.03, 0.08)
            else:
                step_size *= 0.75
                no_improve_count += 1
                if no_improve_count > 5 or step_size < 0.001:
                    break
    
    return best_pts, best_radius

def _polish_step(pts, start_time, max_time=120.0, max_iters=150):
    """
    Final polish: smooth surrogate for max circumradius with vectorized ops.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    beta = 40.0
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        simplices = hull.simplices
        angles = _facet_circumradius_batch(best_pts, simplices)
        
        if len(angles) == 0:
            break
        
        max_angle = np.max(angles)
        if max_angle < 1e-15:
            break
        
        # Soft-max weights
        exp_angles = np.exp(beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Compute circumcenters
        centers = _facet_circumcenters(best_pts, simplices)
        
        # Vectorized Gradient
        grad = np.zeros_like(best_pts)
        
        v1 = best_pts[simplices[:, 0]]
        v2 = best_pts[simplices[:, 1]]
        v3 = best_pts[simplices[:, 2]]
        
        d1 = np.einsum('ij,ij->i', v1, centers)
        d2 = np.einsum('ij,ij->i', v2, centers)
        d3 = np.einsum('ij,ij->i', v3, centers)
        
        dir1 = centers - d1[:, None] * v1
        dir2 = centers - d2[:, None] * v2
        dir3 = centers - d3[:, None] * v3
        
        n1 = np.linalg.norm(dir1, axis=1)
        n2 = np.linalg.norm(dir2, axis=1)
        n3 = np.linalg.norm(dir3, axis=1)
        
        n1[n1 < 1e-10] = 1.0
        n2[n2 < 1e-10] = 1.0
        n3[n3 < 1e-10] = 1.0
        
        w = weights
        wd1 = (w / n1)[:, None] * dir1
        wd2 = (w / n2)[:, None] * dir2
        wd3 = (w / n3)[:, None] * dir3
        
        for i in range(len(simplices)):
            idx1, idx2, idx3 = simplices[i]
            grad[idx1] += wd1[i]
            grad[idx2] += wd2[i]
            grad[idx3] += wd3[i]
        
        # Vectorized repulsion for close points
        diff_mat = best_pts[:, None, :] - best_pts[None, :, :]
        dist_mat = np.linalg.norm(diff_mat, axis=2)
        np.fill_diagonal(dist_mat, 1.0)
        
        close_mask = dist_mat < 0.05
        np.fill_diagonal(close_mask, False)
        
        repulsion = np.zeros_like(best_pts)
        if np.any(close_mask):
            force_mat = np.where(close_mask, 0.01 / (dist_mat + 1e-10), 0.0)
            repulsion = np.sum(force_mat[:, :, None] * diff_mat, axis=1)
        
        total_grad = grad + 0.05 * repulsion
        
        if np.linalg.norm(total_grad) < 1e-15:
            break
        
        # Adaptive step size with backtracking
        step = 0.01
        improved = False
        for _ in range(4):
            new_pts = best_pts + step * total_grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_pts = new_pts
                best_radius = new_radius
                improved = True
                break
            step *= 0.5
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 8:
                break
    
    return best_pts, best_radius

def _lbfgs_smooth(pts, start_time, max_time=120.0, beta=50.0):
    """
    Use L-BFGS-B to optimize the smooth log-sum-exp surrogate based on hull facets.
    """
    deadline = max_time - 15.0
    
    def objective_and_grad(flat_pts):
        if time.time() - start_time > deadline:
            return 0.0, np.zeros_like(flat_pts)
        p = flat_pts.reshape(N, 3)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        try:
            hull = ConvexHull(p)
            simplices = hull.simplices
            angles = _facet_circumradius_batch(p, simplices)
            
            if len(angles) == 0:
                return 0.0, np.zeros_like(flat_pts)
            
            max_angle = np.max(angles)
            # Smooth max: (1/beta) * log(sum(exp(beta * angles)))
            # For stability: max_angle + (1/beta) * log(sum(exp(beta * (angles - max_angle))))
            exp_angles = np.exp(beta * (angles - max_angle))
            sum_exp = np.sum(exp_angles)
            obj_rad = max_angle + np.log(sum_exp) / beta
            obj = np.degrees(obj_rad)
            
            # Gradient
            weights = exp_angles / sum_exp
            
            centers = _facet_circumcenters(p, simplices)
            
            grad = np.zeros_like(p)
            
            v1 = p[simplices[:, 0]]
            v2 = p[simplices[:, 1]]
            v3 = p[simplices[:, 2]]
            
            d1 = np.einsum('ij,ij->i', v1, centers)
            d2 = np.einsum('ij,ij->i', v2, centers)
            d3 = np.einsum('ij,ij->i', v3, centers)
            
            dir1 = centers - d1[:, None] * v1
            dir2 = centers - d2[:, None] * v2
            dir3 = centers - d3[:, None] * v3
            
            n1 = np.linalg.norm(dir1, axis=1)
            n2 = np.linalg.norm(dir2, axis=1)
            n3 = np.linalg.norm(dir3, axis=1)
            
            n1[n1 < 1e-10] = 1.0
            n2[n2 < 1e-10] = 1.0
            n3[n3 < 1e-10] = 1.0
            
            w = weights
            wd1 = (w / n1)[:, None] * dir1
            wd2 = (w / n2)[:, None] * dir2
            wd3 = (w / n3)[:, None] * dir3
            
            for i in range(len(simplices)):
                idx1, idx2, idx3 = simplices[i]
                grad[idx1] += wd1[i]
                grad[idx2] += wd2[i]
                grad[idx3] += wd3[i]
                
            # Convert gradient from radians to degrees
            grad *= np.deg2rad(1.0)
            
            # Project gradient onto tangent space
            dots_p = np.sum(grad * p, axis=1, keepdims=True)
            grad = grad - dots_p * p
            
            return obj, grad.ravel()
        except Exception:
            return 1e10, np.zeros_like(flat_pts)
    
    x0 = pts.ravel()
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                         options={'maxiter': 200, 'maxfun': 200,
                                  'gtol': 1e-6, 'ftol': 1e-10})
        if time.time() - start_time > deadline:
            return pts, _covering_radius(pts)
        new_pts = result.x.reshape(N, 3)
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_radius = _covering_radius(new_pts)
        return new_pts, new_radius
    except Exception:
        return pts, _covering_radius(pts)

def _local_exact_polish(pts, start_time, max_time=120.0):
    """
    Local exact polish: for each facet, try moving its 3 vertices toward the circumcenter
    and accept if the global covering radius improves.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    for _ in range(50):
        if time.time() - start_time > max_time - 8.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        simplices = hull.simplices
        angles = _facet_circumradius_batch(best_pts, simplices)
        
        if len(angles) == 0:
            break
        
        # Sort facets by angle descending
        order = np.argsort(angles)[::-1]
        improved_any = False
        
        for fac_idx in order[:10]:  # Only try top 10 facets
            if time.time() - start_time > max_time - 8.0:
                break
            
            simplex = simplices[fac_idx]
            center = _facet_circumcenters(best_pts, simplices[[fac_idx]])[0]
            
            for step_frac in [0.05, 0.03, 0.01, 0.005]:
                trial_pts = best_pts.copy()
                for idx in simplex:
                    v = trial_pts[idx]
                    direction = center - np.dot(v, center) * v
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-10:
                        direction /= dir_norm
                    trial_pts[idx] = v + step_frac * direction
                    trial_pts[idx] /= np.linalg.norm(trial_pts[idx])
                
                trial_radius = _covering_radius(trial_pts)
                if trial_radius < best_radius - 1e-12:
                    best_pts = trial_pts
                    best_radius = trial_radius
                    improved_any = True
                    break
        
        if not improved_any:
            break
    
    return best_pts, best_radius

def _random_perturb(pts, scale=0.05, seed=None):
    """Add small random perturbation to escape local minima."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * scale
    new_pts = pts + noise
    new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
    return new_pts

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 115.0  # Safety margin before 120s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Lloyd refinement
    elapsed = time.time() - start_time
    if elapsed < max_time - 40.0:
        lloyd_pts, lloyd_radius = _refine_lloyd(pts, start_time, max_time, max_iters=20)
        if lloyd_radius < best_radius:
            best_pts = lloyd_pts
            best_radius = lloyd_radius
    
    # Phase 2: Smooth L-BFGS optimization (Hull-based)
    elapsed = time.time() - start_time
    if elapsed < max_time - 30.0:
        lbfgs_pts, lbfgs_radius = _lbfgs_smooth(best_pts, start_time, max_time, beta=50.0)
        if lbfgs_radius < best_radius:
            best_pts = lbfgs_pts
            best_radius = lbfgs_radius
    
    # Phase 3: Gradient descent on facets
    elapsed = time.time() - start_time
    if elapsed < max_time - 20.0:
        gd_pts, gd_radius = _gradient_descent_facets(best_pts, start_time, max_time)
        if gd_radius < best_radius:
            best_pts = gd_pts
            best_radius = gd_radius
    
    # Phase 4: Local exact polish
    elapsed = time.time() - start_time
    if elapsed < max_time - 10.0:
        polish_pts, polish_radius = _local_exact_polish(best_pts, start_time, max_time)
        if polish_radius < best_radius:
            best_pts = polish_pts
            best_radius = polish_radius
    
    # Phase 5: Try a few random restarts if time permits
    elapsed = time.time() - start_time
    for restart_seed in [42, 123, 777]:
        if time.time() - start_time > max_time - 20.0:
            break
        
        perturbed = _random_perturb(best_pts, scale=0.03, seed=restart_seed)
        
        # Quick Lloyd
        r_pts, r_radius = _refine_lloyd(perturbed, start_time, max_time, max_iters=10)
        
        # Quick L-BFGS
        if time.time() - start_time < max_time - 12.0:
            r_pts2, r_radius2 = _lbfgs_smooth(r_pts, start_time, max_time, beta=60.0)
            if r_radius2 < r_radius:
                r_pts, r_radius = r_pts2, r_radius2
        
        # Quick polish
        if time.time() - start_time < max_time - 8.0:
            r_pts3, r_radius3 = _local_exact_polish(r_pts, start_time, max_time)
            if r_radius3 < r_radius:
                r_pts, r_radius = r_pts3, r_radius
        
        if r_radius < best_radius:
            best_pts = r_pts
            best_radius = r_radius
    
    # Phase 6: Final Lloyd refinement
    elapsed = time.time() - start_time
    if elapsed < max_time - 5.0:
        final_pts, final_radius = _refine_lloyd(best_pts, start_time, max_time, max_iters=10)
        if final_radius < best_radius:
            best_pts = final_pts
            best_radius = final_radius
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
