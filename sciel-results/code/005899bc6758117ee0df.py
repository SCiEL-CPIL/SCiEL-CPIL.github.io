import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 120s limit, leave 10s margin

    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        min_d = np.min(d_opt)
        max_d = np.max(d_opt)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def soft_objective(params, beta=50.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft max: log(sum(exp(beta * d_sq))) / beta
        max_d_sq_soft = np.log(np.sum(np.exp(beta * d_sq))) / beta
        # Soft min: -log(sum(exp(-beta * d_sq))) / beta
        min_d_sq_soft = -np.log(np.sum(np.exp(-beta * d_sq))) / beta
        
        if min_d_sq_soft < 1e-8:
            return 1e10
        
        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)
        
        return (max_d_soft / min_d_soft)**2

    bnds = [(-10, 10)] * (2 * (N - 2))
    
    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # Config 1: Triangular lattice subset
    R = 6
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    pts = np.array(pts)
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    best_pts = pts[initial_subset]
    
    # Find closest pair and normalize
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
    if i0 > i1: i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v > 1e-8:
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        transformed = (best_pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        initial_configs.append(transformed[2:].flatten())
    
    # Config 2: Regular hexagonal pattern with center
    hex_pts = []
    # Center
    hex_pts.append([0.0, 0.0])
    # First ring (6 points)
    for k in range(6):
        angle = 2 * np.pi * k / 6
        hex_pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Second ring (12 points, take 9 to get 16 total)
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 12
        hex_pts.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        if len(hex_pts) == N:
            break
    
    if len(hex_pts) == N:
        hex_arr = np.array(hex_pts)
        # Find closest pair
        diff = hex_arr[:, np.newaxis, :] - hex_arr[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1: i0, i1 = i1, i0
        
        p0 = hex_arr[i0]
        p1 = hex_arr[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v > 1e-8:
            cos_t = v[0] / norm_v
            sin_t = v[1] / norm_v
            R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            transformed = (hex_arr - p0) @ R_mat.T / norm_v
            
            perm = list(range(N))
            perm[0] = i0
            perm[1] = i1
            remaining = [i for i in range(N) if i != i0 and i != i1]
            perm[2:] = remaining
            transformed = transformed[perm]
            initial_configs.append(transformed[2:].flatten())
    
    # Config 3: Random perturbation of triangular
    if len(initial_configs) > 0:
        rng = np.random.RandomState(42)
        perturbed = initial_configs[0].copy()
        perturbed += rng.normal(0, 0.1, size=perturbed.shape)
        initial_configs.append(perturbed)
    
    # Config 4: Square grid
    sq_pts = []
    for i in range(4):
        for j in range(4):
            sq_pts.append([(i - 1.5) * 1.0, (j - 1.5) * 1.0])
    if len(sq_pts) == N:
        sq_arr = np.array(sq_pts)
        diff = sq_arr[:, np.newaxis, :] - sq_arr[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1: i0, i1 = i1, i0
        
        p0 = sq_arr[i0]
        p1 = sq_arr[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v > 1e-8:
            cos_t = v[0] / norm_v
            sin_t = v[1] / norm_v
            R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            transformed = (sq_arr - p0) @ R_mat.T / norm_v
            
            perm = list(range(N))
            perm[0] = i0
            perm[1] = i1
            remaining = [i for i in range(N) if i != i0 and i != i1]
            perm[2:] = remaining
            transformed = transformed[perm]
            initial_configs.append(transformed[2:].flatten())
    
    if not initial_configs:
        # Fallback: random initialization
        rng = np.random.RandomState(0)
        init_params = rng.uniform(-5, 5, size=(2 * (N - 2),))
        initial_configs.append(init_params)
    
    best_params = None
    best_score = float('inf')
    
    # Multi-start optimization
    for start_idx, init_params in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
        
        current_params = init_params.copy()
        current_score = exact_objective(current_params)
        
        if current_score < best_score:
            best_params = current_params.copy()
            best_score = current_score
        
        # Stage 1: Smooth surrogate with beta=50
        try:
            res = minimize(soft_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
            if res.fun < 1e9:
                exact_score = exact_objective(res.x)
                if exact_score < current_score:
                    current_params = res.x
                    current_score = exact_score
        except Exception:
            pass
        
        # Stage 2: Smooth surrogate with beta=30 for escape
        if time.time() < deadline - 10:
            def soft_beta30(params):
                return soft_objective(params, beta=30.0)
            try:
                res = minimize(soft_beta30, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < current_score:
                        current_params = res.x
                        current_score = exact_score
            except Exception:
                pass
        
        # Stage 3: Exact objective
        if time.time() < deadline - 8:
            try:
                res = minimize(exact_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
                if res.fun < current_score:
                    current_params = res.x
                    current_score = res.fun
            except Exception:
                pass
        
        # Stage 4: High-precision polish
        if time.time() < deadline - 5:
            try:
                res = minimize(exact_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
                if res.fun < current_score:
                    current_params = res.x
                    current_score = res.fun
            except Exception:
                pass
        
        if current_score < best_score:
            best_params = current_params.copy()
            best_score = current_score
    
    # Final global polish on best solution
    if best_params is not None and time.time() < deadline - 3:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 20})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
