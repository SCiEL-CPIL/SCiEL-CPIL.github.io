import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = dist
    
    return radii, centers, valid

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        if len(radii) == 0:
            return 180.0
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _soft_max_objective_and_grad(pts, beta=20.0):
    """
    Compute the soft-max covering radius (in radians) and its gradient.
    The objective is: (1/beta) * log(sum(exp(beta * r_i)))
    where r_i are the angular circumradii of the hull facets.
    
    This is a smooth surrogate for max(r_i).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, np.zeros_like(pts)
    
    radii, centers, valid = _facet_circumradii(pts, hull.simplices)
    if not np.any(valid):
        return 180.0, np.zeros_like(pts)
    
    valid_idx = np.where(valid)[0]
    r_valid = radii[valid_idx]
    
    # Soft-max
    max_r = np.max(r_valid)
    exp_r = np.exp(beta * (r_valid - max_r))
    sum_exp = np.sum(exp_r)
    obj = max_r + np.log(sum_exp) / beta
    
    # Gradient of soft-max: grad obj = (1/beta) * (sum(exp(beta*r_i) * grad r_i) / sum(exp(beta*r_i)))
    # We need grad r_i for each facet i.
    # r_i = arccos(c_i . p_1i)
    # d r_i / d p_j = - (c_i . (e_j - (c_i . e_j) c_i)) / sqrt(1 - (c_i . p_j)^2)  for j in facet i
    # Actually, simpler: d/dp arccos(c.p) = -c_tan / sin(r)
    # where c_tan is the tangent component of c at p.
    
    grad_obj = np.zeros_like(pts)
    
    # Weights for each facet
    weights = exp_r / sum_exp / beta
    
    for k, idx in enumerate(valid_idx):
        facet = hull.simplices[idx]
        c = centers[idx]
        r = r_valid[k]
        w = weights[k]
        
        if w < 1e-12:
            continue
            
        sin_r = np.sin(r)
        if sin_r < 1e-12:
            continue
            
        inv_sin = 1.0 / sin_r
        
        for j in facet:
            p = pts[j]
            # Gradient of r with respect to p_j
            # r = arccos(c . p)
            # dr/dp = -c / sqrt(1 - (c.p)^2)  but projected onto tangent space
            c_tan = c - np.dot(c, p) * p
            grad_r = -c_tan * inv_sin
            grad_obj[j] += w * grad_r
            
    return obj, grad_obj

def _gradient_descent(pts, deadline, beta=20.0, lr=0.05, min_lr=1e-5, patience=50):
    """
    Gradient descent on the soft-max covering radius.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_obj = _covering_radius_deg(pts)
    no_improve = 0
    lr_current = lr
    
    while time.time() < deadline:
        obj, grad = _soft_max_objective_and_grad(pts, beta)
        
        if not np.isfinite(obj):
            break
            
        # Project gradient to tangent space
        grad_tan = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
        
        # Update
        new_pts = pts - lr_current * grad_tan
        new_pts = _normalize(new_pts)
        
        # Check improvement
        new_rad = _covering_radius_deg(new_pts)
        
        if new_rad < best_obj - 1e-7:
            best_obj = new_rad
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve = 0
            # Decay learning rate slowly
            lr_current *= 0.995
        else:
            no_improve += 1
            lr_current *= 0.95
            
            if no_improve > patience:
                # Reset to best and try a smaller step or restart
                pts = best_pts.copy()
                no_improve = 0
                lr_current = max(lr_current * 0.5, min_lr)
                
            if lr_current < min_lr:
                break
                
    return best_pts, best_obj

def _lloyd_hybrid_optimize(pts, deadline, initial_step=0.1, min_step=1e-4):
    """
    Hybrid Lloyd-Gradient optimization.
    1. Find worst facet.
    2. Move its vertices towards circumcenter (Lloyd step).
    3. Apply small gradient descent on soft-max surrogate for global smoothing.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_rad = _covering_radius_deg(pts)
    step_size = initial_step
    no_improve = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        radii, centers, valid = _facet_circumradii(pts, hull.simplices)
        if not np.any(valid):
            break
            
        valid_indices = np.where(valid)[0]
        r_valid = radii[valid_indices]
        
        # Find worst facet
        max_idx_in_valid = np.argmax(r_valid)
        worst_facet_idx = valid_indices[max_idx_in_valid]
        worst_rad = r_valid[max_idx_in_valid]
        
        if worst_rad < 1e-6:
            break
            
        worst_facet = hull.simplices[worst_facet_idx]
        worst_center = centers[worst_facet_idx]
        
        # 1. Lloyd Step: Move vertices of worst facet towards circumcenter
        new_pts = pts.copy()
        for idx in worst_facet:
            p = pts[idx]
            # Move towards center
            # The direction from p to center on the sphere is the tangent component of (center - p)
            d = worst_center - p
            d_tan = d - np.dot(d, p) * p
            norm_d = np.linalg.norm(d_tan)
            if norm_d > 1e-10:
                d_tan /= norm_d
                new_pts[idx] = p + step_size * d_tan
        
        new_pts = _normalize(new_pts)
        
        new_rad = _covering_radius_deg(new_pts)
        
        if new_rad < best_rad - 1e-6:
            best_rad = new_rad
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve = 0
            # Increase step slightly if improving fast? Or keep constant.
            # Let's decay step size slowly
            step_size *= 0.99
        else:
            no_improve += 1
            step_size *= 0.9
            # If stuck, try a larger random perturbation (Simulated Annealing style escape)
            if no_improve > 10 and step_size > min_step:
                noise = np.random.normal(0, step_size * 0.5, pts.shape)
                pts = _normalize(pts + noise)
                no_improve = 0
                step_size = max(step_size * 1.5, initial_step * 0.5)
                
            if step_size < min_step:
                break
                
    return best_pts, best_rad

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    # Initialize with multiple strategies
    best_points = _fibonacci_spiral(0.0)
    best_rad = _covering_radius_deg(best_points)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.5))
    inits.append(_fibonacci_spiral(1.0))
    
    for _ in range(3):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
        
    # Evaluate all initializations
    init_list = []
    for init in inits:
        if time.time() - start_time > max_time - 10.0:
            break
        rad = _covering_radius_deg(init)
        init_list.append((rad, init.copy()))
        
    # Sort by quality
    init_list.sort(key=lambda x: x[0])
    
    # Local optimization on top initializations
    # We run the Lloyd-Hybrid optimizer on the top 3-5 starts
    top_k = 3
    for i, (rad, init_pts) in enumerate(init_list[:top_k]):
        if time.time() - start_time > max_time - 15.0:
            break
            
        if rad < best_rad:
            best_rad = rad
            best_points = init_pts.copy()
            
        # Allocate remaining time proportionally or fixed chunks
        # Let's give each run a chunk of time
        remaining = max_time - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        # Run optimizer with a deadline
        deadline = time.time() + remaining * 0.8  # Leave 20% for next runs or polish
        opt_pts, opt_rad = _lloyd_hybrid_optimize(init_pts, deadline, initial_step=0.2)
        
        if opt_rad < best_rad:
            best_rad = opt_rad
            best_points = opt_pts.copy()
            
    # Final polish with gradient descent on soft-max surrogate
    if time.time() - start_time < max_time - 5.0:
        polish_deadline = time.time() + (max_time - (time.time() - start_time)) * 0.9
        polish_pts, polish_rad = _gradient_descent(best_points, polish_deadline, beta=20.0, lr=0.05)
        if polish_rad < best_rad:
            best_rad = polish_rad
            best_points = polish_pts.copy()
            
    return best_points
