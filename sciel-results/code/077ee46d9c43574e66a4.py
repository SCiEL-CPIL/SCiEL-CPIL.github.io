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
    
    # 3. Define smooth objective and gradient for L-BFGS using Soft-Min Surrogate
    # This approach uses a differentiable approximation of the min function to handle
    # the nonsmoothness of the covering radius objective more robustly.
    
    def make_obj_grad(beta):
        def objective_and_grad(P_flat):
            P = P_flat.reshape(-1, 3)
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            P = P / norms
            
            dots = samples_opt @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            
            # Soft-min calculation
            min_dists = np.min(dists, axis=1, keepdims=True)
            scaled_dists = dists - min_dists
            exp_terms = np.exp(-beta * scaled_dists)
            sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
            weights = exp_terms / sum_exp
            
            log_sum_exp_scaled = np.log(sum_exp)
            soft_mins = min_dists.flatten() - (1.0 / beta) * log_sum_exp_scaled.flatten()
            loss = np.sum(soft_mins)
            
            # Gradient
            denom = np.sqrt(np.maximum(1e-12, 1.0 - dots**2))
            term = -weights / denom
            grad_P_ambient = term.T @ samples_opt
            
            grad_dot_P = np.sum(grad_P_ambient * P, axis=1, keepdims=True)
            grad_tangent = grad_P_ambient - grad_dot_P * P
            
            norms_flat = np.linalg.norm(P_flat.reshape(-1, 3), axis=1, keepdims=True)
            norms_flat[norms_flat < 1e-15] = 1.0
            grad_P_raw = grad_tangent / norms_flat
            
            return loss, grad_P_raw.flatten()
        return objective_and_grad

    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 30.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")

    try:
        # Phase 1: Coarse optimization with lower beta (smoother landscape)
        obj_grad_1 = make_obj_grad(20.0)
        res1 = minimize(
            obj_grad_1,
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5, 'disp': False},
            callback=time_callback
        )
        if res1.x is not None:
            P = res1.x.reshape(-1, 3)
            P = _normalize(P)
            
        if not time_check[0]:
            # Phase 2: Fine optimization with higher beta (sharper approximation)
            obj_grad_2 = make_obj_grad(100.0)
            res2 = minimize(
                obj_grad_2,
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False},
                callback=time_callback
            )
            if res2.x is not None:
                P = res2.x.reshape(-1, 3)
                P = _normalize(P)
    except Exception:
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
