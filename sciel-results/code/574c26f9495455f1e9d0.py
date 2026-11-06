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

    best_P = None
    best_score = 180.0

    # Multiple restarts to avoid local minima
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # 1. Initialization with varying noise/rotation
        # Use different seeds for noise
        seed = 42 + restart * 100
        # Add a small random rotation to the base Fibonacci sphere to break symmetry
        P = _fibonacci_sphere(N, seed_noise=0.05, seed=seed)
        
        # Random rotation
        rng = np.random.default_rng(seed)
        # Generate a random rotation matrix
        # Using QR decomposition of a random matrix
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        # Ensure proper rotation (det=1)
        d = np.sign(np.linalg.det(Q))
        Q[:, -1] *= d
        P = P @ Q.T
        P = _normalize(P)

        # 2. Prepare samples for optimization
        S_opt = 8000
        samples_opt = _fibonacci_sphere(S_opt)
        
        # 3. Define smooth objective and gradient for L-BFGS
        lambda_val = 12.0
        
        def objective_and_grad(P_flat):
            P = P_flat.reshape(-1, 3)
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            P = P / norms
            
            # Compute distances from samples to all points
            dots = samples_opt @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            
            min_dists = np.min(dists, axis=1)
            nearest_idx = np.argmin(dists, axis=1)
            
            # Loss
            loss = np.sum(np.exp(lambda_val * min_dists))
            
            # Gradient
            grad_P = np.zeros_like(P)
            
            weights = lambda_val * np.exp(lambda_val * min_dists)
            
            dots_nearest = dots[np.arange(S_opt), nearest_idx]
            denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
            
            grad_factor = -weights / denom
            
            np.add.at(grad_P, nearest_idx, grad_factor[:, None] * samples_opt)
            
            grad_dot_P = np.sum(grad_P * P, axis=1, keepdims=True)
            grad_tangent = grad_P - grad_dot_P * P
            
            norms_flat = np.linalg.norm(P_flat.reshape(-1, 3), axis=1, keepdims=True)
            norms_flat[norms_flat < 1e-15] = 1.0
            
            grad_P_raw = grad_tangent / norms_flat
            
            return loss, grad_P_raw.flatten()
        
        # Run L-BFGS-B
        time_check = [False]
        def time_callback(xk):
            if time.time() > deadline - 25.0:
                time_check[0] = True
                raise ValueError("Time limit approaching")
        
        try:
            res = minimize(
                objective_and_grad,
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False},
                callback=time_callback
            )
            if res.x is not None:
                P = res.x.reshape(-1, 3)
                P = _normalize(P)
        except Exception:
            pass

        # 4. Polish with high-resolution greedy local search
        # Use a smaller sample set for speed during the greedy loop
        S_polish = 5000
        samples_polish = _fibonacci_sphere(S_polish)
        
        def get_min_dists(P):
            dots = samples_polish @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            nearest_idx = np.argmin(dists, axis=1)
            return min_dists, nearest_idx
        
        min_dists, _ = get_min_dists(P)
        current_max_dist = np.max(min_dists)
        
        rng = np.random.default_rng(42 + restart)
        
        # Greedy polish loop
        # We iterate until time runs out or no improvement for a while
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < deadline - 10.0 and no_improve_count < max_no_improve:
            min_dists, nearest_idx = get_min_dists(P)
            worst_idx = np.argmax(min_dists)
            worst_sample = samples_polish[worst_idx]
            
            p_idx = nearest_idx[worst_idx]
            p_old = P[p_idx].copy()
            
            # Move p_old towards worst_sample
            proj = np.dot(p_old, worst_sample)
            grad_dir = worst_sample - proj * p_old
            grad_norm = np.linalg.norm(grad_dir)
            
            if grad_norm < 1e-6:
                no_improve_count += 1
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
            best_step = 0.0
            best_new_max = current_max_dist
            best_p = p_old.copy()
            
            # Try a few step sizes
            for step_size in [0.2, 0.1, 0.05, 0.02, 0.01]:
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
                no_improve_count = 0
            else:
                no_improve_count += 1

        # Evaluate the current best candidate
        # Use a high resolution set for the final score estimation
        S_eval = 20000
        samples_eval = _fibonacci_sphere(S_eval)
        dots = samples_eval @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        score = np.max(min_dists)
        
        if score < best_score:
            best_score = score
            best_P = P.copy()

    # If no valid result was found, return the initial Fibonacci sphere
    if best_P is None:
        best_P = _fibonacci_sphere(N, seed_noise=0.01)
        
    return best_P
# EVOLVE-BLOCK-END
