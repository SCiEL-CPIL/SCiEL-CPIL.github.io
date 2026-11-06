"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) >= 3:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _get_min_dists(P, samples):
    dots = samples @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest_idx = np.argmin(dists, axis=1)
    return min_dists, nearest_idx

def _hull_circumradius(P):
    """Compute the maximum circumradius of hull facets (the true covering radius)."""
    P = _normalize(P)
    try:
        hull = ConvexHull(P)
    except Exception:
        return np.pi  # fallback: 180 degrees in radians
    
    max_r = 0.0
    for simplex in hull.simplices:
        verts = P[simplex]
        if len(verts) < 3:
            continue
        # For each facet, compute the circumcenter of the circumscribed sphere
        # The facet's circumradius on the unit sphere is the angular distance
        # from the facet center (normal direction) to each vertex
        
        v0, v1, v2 = verts[0], verts[1], verts[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        # Ensure normal points outward
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # Circumradius is the max angular distance from normal to any vertex
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_r:
                max_r = ang
    
    return max_r

def _smooth_max_surrogate(P, samples, temperature=0.05):
    """
    Compute a smooth approximation of the max distance using log-sum-exp.
    Returns (value, gradient) where gradient is w.r.t. the flattened P.
    """
    dots = samples @ P.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    dists = np.arccos(dots)  # (S, N)
    min_dists = np.min(dists, axis=1)  # (S,)
    
    # Log-sum-exp for smooth max
    t = 1.0 / temperature
    max_d = np.max(min_dists)
    exp_shifted = np.exp(t * (min_dists - max_d))
    log_sum = np.log(np.sum(exp_shifted))
    value = max_d + log_sum / t
    
    # Gradient w.r.t. min_dists
    weights = exp_shifted / np.sum(exp_shifted)  # (S,)
    
    # Gradient w.r.t. P: for each sample s, the nearest point j
    nearest_idx = np.argmin(dists, axis=1)  # (S,)
    grad_P = np.zeros((N, 3))
    
    # d(min_dists)/d(P[j]) for the nearest point j to sample s
    # min_dists[s] = arccos(dots[s, j]) where j = nearest_idx[s]
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # d(dots[s,j])/d(P[j]) = -samples[s]
    # So d(min_dists[s])/d(P[j]) = samples[s] / sqrt(1 - dots[s,j]^2)
    
    for s in range(len(nearest_idx)):
        j = nearest_idx[s]
        w = weights[s]
        if w < 1e-15:
            continue
        d_val = dots[s, j]
        denom = np.sqrt(max(1.0 - d_val * d_val, 1e-15))
        grad_P[j] += w * samples[s] / denom
    
    return value, grad_P

def _riemannian_gradient_step(P, grad, step_size):
    """
    Perform a Riemannian gradient step on the sphere.
    P: (N, 3) array of unit vectors
    grad: (N, 3) array of tangent-space gradients
    Returns updated P.
    """
    # Project gradient to tangent space
    grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot_P * P
    
    # Exponential map on the sphere: exp_P(v) = P*cos(|v|) + (v/|v|)*sin(|v|)
    v = -step_size * grad_tangent
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    
    # Handle small norms
    small = v_norm < 1e-10
    v_norm[small] = 1.0
    
    cos_term = np.cos(v_norm)
    sin_term = np.sin(v_norm) / v_norm
    
    P_new = P * cos_term + v * sin_term
    P_new = _normalize(P_new)
    
    return P_new

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Initialization with slight noise for symmetry breaking
    P = _fibonacci_sphere(N, seed_noise=0.01)
    
    # 2. Generate samples for optimization
    S_opt = 15000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # Phase 1: Riemannian L-BFGS-B optimization on the smooth max surrogate
    # We use the log-sum-exp smooth max of the min-distances to samples
    
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt, temperature=0.05)
        
        # Project gradient to tangent space
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        # For L-BFGS-B in Euclidean space, we need to account for the constraint
        # Use a penalty-free approach: the optimizer will handle it via the
        # normalization we do inside the objective
        return val, grad_tangent.flatten()
    
    try:
        # Use L-BFGS-B with the Riemannian gradient projected to Euclidean
        # The key insight: since we normalize inside the objective, the 
        # effective gradient is the tangent-space gradient
        res = minimize(
            objective_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 40.0:
        # Skip to final polish
        pass
    else:
        # Phase 2: Refine with adaptive temperature (annealing)
        for temp in [0.03, 0.01, 0.005]:
            if time.time() > deadline - 30.0:
                break
            
            def obj_temp(P_flat):
                P = P_flat.reshape(-1, 3)
                P = _normalize(P)
                val, grad = _smooth_max_surrogate(P, samples_opt, temperature=temp)
                grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
                grad_tangent = grad - grad_dot_P * P
                return val, grad_tangent.flatten()
            
            try:
                res = minimize(
                    obj_temp,
                    P.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                )
                if res.x is not None:
                    P = res.x.reshape(-1, 3)
                    P = _normalize(P)
            except Exception:
                pass
    
    # Phase 3: Greedy local search with high-resolution samples
    S_polish = 40000
    samples_polish = _fibonacci_sphere(S_polish)
    
    min_dists, nearest_idx = _get_min_dists(P, samples_polish)
    current_max_dist = np.max(min_dists)
    
    rng = np.random.default_rng(123)
    
    # Iterative improvement: move the point nearest to the worst sample
    max_iters = 300
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        min_dists, nearest_idx = _get_min_dists(P, samples_polish)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples_polish[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        # Gradient direction: move p_old toward worst_sample (tangent)
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-6:
            # Try moving away from the nearest neighbor instead
            # Find the second nearest point
            dists_all = np.arccos(np.clip(samples_polish[worst_idx] @ P.T, -1.0, 1.0))
            order = np.argsort(dists_all)
            if len(order) > 1:
                second_nearest = P[order[1]]
                proj2 = np.dot(p_old, second_nearest)
                grad_dir = p_old - proj2 * second_nearest
                grad_norm = np.linalg.norm(grad_dir)
                if grad_norm < 1e-6:
                    continue
                grad_dir = grad_dir / grad_norm
            else:
                continue
        else:
            grad_dir = grad_dir / grad_norm
        
        # Add small random component
        random_dir = rng.standard_normal(3)
        random_dir = random_dir - np.dot(random_dir, p_old) * p_old
        random_norm = np.linalg.norm(random_dir)
        if random_norm > 1e-6:
            random_dir = random_dir / random_norm
        else:
            random_dir = np.zeros(3)
        
        # Line search
        best_step = 0.0
        best_new_max = current_max_dist
        best_p = p_old.copy()
        
        for step_size in [0.08, 0.04, 0.02, 0.01, 0.005]:
            new_p = p_old + step_size * (0.85 * grad_dir + 0.15 * random_dir)
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = _get_min_dists(P, samples_polish)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max_dist = best_new_max
        else:
            P[p_idx] = p_old
    
    # Phase 4: Final verification and cleanup
    P = _normalize(P)
    
    return P

# EVOLVE-BLOCK-END
