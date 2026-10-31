import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145

    # --- Helper Functions ---
    
    def compute_ratio(pts):
        """Compute squared max/min distance ratio."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def normalize_and_align(pts):
        """Normalize points so closest pair is at (0,0) and (1,0)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return pts, pts[2:].flatten(), 0, 1
        
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed, transformed[2:].flatten(), i0, i1

    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        return compute_ratio(pts_opt)

    def soft_objective(params, beta=50.0):
        """Smooth surrogate using log-sum-exp."""
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_val = np.min(d_sq)
        max_val = np.max(d_sq)
        
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        # Soft min: -(1/beta) * log(sum(exp(-beta * x)))
        
        # Numerical stability
        soft_max_log = max_val + np.log(np.sum(np.exp(beta * (d_sq - max_val))))
        soft_min_log = min_val + np.log(np.sum(np.exp(-beta * (d_sq - min_val))))
        
        max_d_soft = np.sqrt(soft_max_log / beta)
        min_d_soft = np.sqrt(-soft_min_log / beta)
        
        if min_d_soft < 1e-10:
            return 1e10
        
        return (max_d_soft / min_d_soft)**2

    # --- Phase 1: Discrete Search on Triangular Lattice ---
    
    # Generate a large triangular lattice
    R = 10
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    
    # Initial guess: 16 closest points to origin
    dists_origin = np.sum(lattice_pts**2, axis=1)
    sorted_idx = np.argsort(dists_origin)
    
    # Pool of candidates: top 200
    pool_size = 200
    candidate_indices = sorted_idx[:pool_size]
    
    # Simulated Annealing on the subset selection
    current_indices = sorted_idx[:N].copy()
    current_score = compute_ratio(lattice_pts[current_indices])
    
    best_indices = current_indices.copy()
    best_score = current_score
    
    T = 1.0
    T_min = 1e-4
    cooling_rate = 0.95
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    max_sa_time = 10.0
    sa_start = time.time()
    
    while T > T_min and (time.time() - sa_start) < max_sa_time and time.time() < deadline - 30:
        # Pick a random point in subset and a random point outside
        i_idx = np.random.randint(N)
        i = current_indices[i_idx]
        
        if not outside_candidates:
            break
            
        j_idx = np.random.randint(len(outside_candidates))
        j = outside_candidates[j_idx]
        
        new_indices = current_indices.copy()
        new_indices[i_idx] = j
        new_score = compute_ratio(lattice_pts[new_indices])
        
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            current_indices = new_indices
            current_score = new_score
            
            # Update sets
            in_subset = set(current_indices)
            outside_candidates = [k for k in candidate_indices if k not in in_subset]
            
            if current_score < best_score:
                best_score = current_score
                best_indices = current_indices.copy()
                
        T *= cooling_rate
        
    best_lattice_pts = lattice_pts[best_indices]
    
    # --- Phase 2: Continuous Optimization ---
    
    best_params = None
    best_final_score = 1e10
    
    # Initial configurations for continuous optimization
    configs = []
    
    # Config 1: Best lattice subset
    configs.append(best_lattice_pts)
    
    # Config 2: 16 closest to origin (in case SA got stuck)
    configs.append(lattice_pts[sorted_idx[:N]])
    
    # Config 3: Regular 4x4 grid (centered)
    grid_pts = np.array([(i - 1.5, j - 1.5) for i in range(4) for j in range(4)])
    configs.append(grid_pts)
    
    # Config 4: Two concentric circles
    circle_pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        circle_pts.append((0.5 * np.cos(angle), 0.5 * np.sin(angle)))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        circle_pts.append((1.5 * np.cos(angle), 1.5 * np.sin(angle)))
    configs.append(np.array(circle_pts))
    
    for config in configs:
        if time.time() > deadline - 20:
            break
            
        aligned, params, i0, i1 = normalize_and_align(config)
        current_score = exact_objective(params)
        
        if current_score < best_final_score:
            best_final_score = current_score
            best_params = params.copy()
            
        bnds = [(-10, 10)] * (2 * (N - 2))
        
        # Phase 2a: Differential Evolution (Global)
        # Only run if we have time. DE is slow.
        if time.time() < deadline - 40:
            try:
                # Use a small population and few generations to keep it fast
                de_res = differential_evolution(
                    exact_objective, 
                    bounds=bnds, 
                    seed=42, 
                    maxiter=50, 
                    popsize=10, 
                    tol=1e-6,
                    polish=False # We will polish manually
                )
                if de_res.fun < best_final_score:
                    best_final_score = de_res.fun
                    best_params = de_res.x.copy()
                    params = de_res.x.copy()
            except Exception:
                pass
        
        # Phase 2b: L-BFGS-B on Soft Objective
        if time.time() < deadline - 15:
            try:
                res = minimize(soft_objective, params, method='L-BFGS-B', bounds=bnds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < 1e9:
                    exact_s = exact_objective(res.x)
                    if exact_s < best_final_score:
                        best_final_score = exact_s
                        best_params = res.x.copy()
                        params = res.x.copy()
            except Exception:
                pass
        
        # Phase 2c: L-BFGS-B on Exact Objective
        if time.time() < deadline - 10:
            try:
                res = minimize(exact_objective, params, method='L-BFGS-B', bounds=bnds,
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < best_final_score:
                    best_final_score = res.fun
                    best_params = res.x.copy()
                    params = res.x.copy()
            except Exception:
                pass
        
        # Phase 2d: Nelder-Mead Polish
        if time.time() < deadline - 5:
            try:
                res = minimize(exact_objective, params, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < best_final_score:
                    best_final_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish on the absolute best
    if time.time() < deadline - 2:
        try:
            res = minimize(exact_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_final_score:
                best_final_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
