"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
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
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) == 3:
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        elif len(verts) == 4:
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization
    P = _fibonacci_sphere(N, seed_noise=0.01)
    
    # 2. Prepare samples for optimization
    # Use a moderate resolution for the global optimization phase
    S_opt = 10000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 3. Define smooth objective and gradient for L-BFGS
    # Objective: Sum of exp(lambda * min_dist)
    # This penalizes large gaps exponentially, effectively minimizing the max distance
    lambda_val = 10.0
    
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        P = P / norms
        
        # Compute distances from samples to all points
        # dots (S, N)
        dots = samples_opt @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        # Squared chordal distance is proportional to 2 - 2*dots
        # Angular distance theta ~ sqrt(2 - 2*cos(theta)) for small theta
        # Let's use angular distance directly for better scaling
        dists = np.arccos(dots) # (S, N)
        
        # Min distance for each sample
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        # Loss
        loss = np.sum(np.exp(lambda_val * min_dists))
        
        # Gradient
        grad_P = np.zeros_like(P)
        
        # Weights for each sample
        # d/dP_i (min_dist) is 0 unless i is the nearest neighbor
        # If i is nearest, d/dP_i (dist) = (P_i - s) / |P_i - s| ... wait
        # dist = arccos(P_i . s)
        # d dist / d P_i = - s / sqrt(1 - (P_i . s)^2)
        # But we are minimizing dist.
        # Let's use the chain rule carefully.
        
        # Actually, simpler gradient for soft-min:
        # If we use loss = sum(exp(lambda * min_dists))
        # d loss / d min_dists[s] = lambda * exp(lambda * min_dists[s])
        # d min_dists[s] / d P_i = d dist(s, P_i) / d P_i if i == nearest_idx[s] else 0
        # d dist(s, P_i) / d P_i = -s / sqrt(1 - (s.P_i)^2)
        
        weights = lambda_val * np.exp(lambda_val * min_dists)
        
        # For each sample s, the gradient contribution is:
        # weights[s] * (-s / sqrt(1 - (s.P_nearest[s])^2))
        
        # Let's compute the denominator
        # dots_nearest = dots[np.arange(S_opt), nearest_idx]
        # denom = sqrt(1 - dots_nearest^2)
        
        dots_nearest = dots[np.arange(S_opt), nearest_idx]
        denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
        
        # grad_factor = - weights / denom
        grad_factor = -weights / denom
        
        # Accumulate gradient
        # grad_P[nearest_idx[s]] += grad_factor[s] * samples_opt[s]
        # Vectorized:
        np.add.at(grad_P, nearest_idx, grad_factor[:, None] * samples_opt)
        
        # Project gradient to tangent space of sphere
        # grad_P is in R^3 space, but P is constrained to sphere.
        # The gradient in the ambient space is grad_P.
        # The gradient on the manifold is grad_P - (grad_P . P) P
        
        grad_dot_P = np.sum(grad_P * P, axis=1, keepdims=True)
        grad_tangent = grad_P - grad_dot_P * P
        
        # Map back to raw parameters P_raw
        # P = P_raw / norm(P_raw)
        # d P / d P_raw = (I - P P^T) / norm(P_raw)
        # grad_P_raw = (grad_tangent) * norm(P_raw)
        
        # Wait, if P = P_raw / ||P_raw||, then d P_i / d P_raw_j = (delta_ij ||P|| - P_i P_j) / ||P||^2
        # grad_P_raw = J^T grad_P
        # J = (I - P P^T) / ||P_raw||
        # Since ||P||=1, ||P_raw|| is just the norm of the unnormalized vector.
        # In our objective, we normalized P. So P in the function is the normalized version.
        # The input to objective is P_flat (unnormalized).
        # So we need d L / d P_flat.
        # L(P_norm) where P_norm = P_flat / ||P_flat||
        # d L / d P_flat = (I - P_norm P_norm^T) * grad_L_dP_norm * ||P_flat||^{-1} ?
        # Actually, standard result: if y = x/||x||, then dy/dx = (I - yy^T)/||x||.
        # So grad_x = ((I - yy^T) grad_y) / ||x||
        
        # We have grad_y = grad_tangent (which is already (I - P P^T) grad_P_ambient)
        # Wait, grad_P calculated above is dL/dP_ambient.
        # So grad_tangent = (I - P P^T) grad_P.
        # Then grad_P_raw = grad_tangent / ||P_flat||
        
        norms_flat = np.linalg.norm(P_flat.reshape(-1, 3), axis=1, keepdims=True)
        norms_flat[norms_flat < 1e-15] = 1.0
        
        grad_P_raw = grad_tangent / norms_flat
        
        return loss, grad_P_raw.flatten()
    
    # Run L-BFGS-B
    # We use a callback to check time
    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 20.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")
    
    try:
        res = minimize(
            objective_and_grad,
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False},
            callback=time_callback
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        # If it fails, keep current P
        pass

    # 4. Polish with high-resolution greedy local search
    S_polish = 50000
    samples_polish = _fibonacci_sphere(S_polish)
    
    # Helper to get min dist and nearest idx
    def get_min_dists(P):
        dots = samples_polish @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        return min_dists, nearest_idx
    
    min_dists, _ = get_min_dists(P)
    current_max_dist = np.max(min_dists)
    
    rng = np.random.default_rng(42)
    
    # Greedy polish loop
    # Move points towards the worst covered sample
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
            
        min_dists, nearest_idx = get_min_dists(P)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples_polish[worst_idx]
        
        # Find nearest point to worst sample
        # Actually, nearest_idx[worst_idx] gives the nearest point to the worst sample
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        # Move p_old towards worst_sample
        # Tangent direction
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-6:
            continue
            
        grad_dir = grad_dir / grad_norm
        
        # Add small random noise to escape local minima
        random_dir = rng.standard_normal(3)
        random_dir = random_dir - np.dot(random_dir, p_old) * p_old
        random_norm = np.linalg.norm(random_dir)
        if random_norm > 1e-6:
            random_dir = random_dir / random_norm
        else:
            random_dir = np.zeros(3)
            
        # Adaptive step size
        # Try a few step sizes
        best_step = 0.0
        best_new_max = current_max_dist
        
        for step_size in [0.1, 0.05, 0.01, 0.005]:
            new_p = p_old + step_size * (0.8 * grad_dir + 0.2 * random_dir)
            new_p = _normalize(new_p.reshape(1,3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = get_min_dists(P)
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
            # If no improvement, revert (already done in loop)
            P[p_idx] = p_old
            # Optionally, if stuck for a while, break or reduce noise
            pass

    return P
# EVOLVE-BLOCK-END
