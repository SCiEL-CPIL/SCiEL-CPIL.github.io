import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 148  # 150s limit, leave 2s margin

    # --- Objective Functions ---
    
    def exact_objective(params):
        """Exact objective: (max_d / min_d)^2. params are 2*(N-2) free coords."""
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        # Vectorized distance calculation
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        min_d = np.min(d_opt)
        max_d = np.max(d_opt)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def smooth_objective(params, beta=300.0):
        """
        Smooth surrogate using log-sum-exp for max and min.
        soft_max = (1/beta) * log(sum(exp(beta * d)))
        soft_min = -(1/beta) * log(sum(exp(-beta * d)))
        """
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        # Clamp to avoid log(0) or overflow
        d_opt = np.clip(d_opt, 1e-10, 1e6)
        
        # Stable log-sum-exp implementation
        max_d = np.max(d_opt)
        min_d = np.min(d_opt)
        
        # soft_max
        log_sum_exp_max = max_d + np.log(np.sum(np.exp(beta * (d_opt - max_d))))
        soft_max = (1.0 / beta) * log_sum_exp_max
        
        # soft_min
        log_sum_exp_min = -min_d + np.log(np.sum(np.exp(-beta * (d_opt - min_d))))
        soft_min = -(1.0 / beta) * log_sum_exp_min
        
        if soft_min < 1e-8:
            return 1e10
        return (soft_max / soft_min)**2

    def normalize_points(pts_in):
        """
        Normalize points: fix p0->(0,0), p1->(1,0) where (p0,p1) is the closest pair.
        Returns (init_params, perm).
        """
        diff = pts_in[:, np.newaxis, :] - pts_in[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1: i0, i1 = i1, i0
        
        p0 = pts_in[i0]
        p1 = pts_in[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            # Fallback: use first two points
            i0, i1 = 0, 1
            p0 = pts_in[0]
            p1 = pts_in[1]
            v = p1 - p0
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                return None, None
        
        # Align p0->(0,0), p1->(1,0)
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts_in - p0) @ R_mat.T / norm_v
        
        # Reorder
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        return init_params, perm

    # --- Initial Configurations ---
    
    initial_configs = []
    
    # Config 1: Triangular lattice cluster
    R = 6
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    dists_from_origin = np.sum(lattice_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_configs.append(lattice_pts[sorted_indices[:N]])
    
    # Config 2: Circular pattern (16 points on a circle)
    circle_pts = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        circle_pts[i] = [np.cos(angle), np.sin(angle)]
    initial_configs.append(circle_pts)
    
    # Config 3: Two concentric circles (8 inner, 8 outer)
    circle2_pts = np.zeros((N, 2))
    for i in range(8):
        angle = 2 * np.pi * i / 8
        circle2_pts[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        circle2_pts[8 + i] = [1.0 * np.cos(angle), 1.0 * np.sin(angle)]
    initial_configs.append(circle2_pts)
    
    # Config 4: Square grid 4x4
    grid_pts = np.zeros((N, 2))
    for i in range(4):
        for j in range(4):
            grid_pts[4*i + j] = [i, j]
    initial_configs.append(grid_pts)
    
    # Config 5: Triangular arrangement (rows of 4)
    tri_pts = np.zeros((N, 2))
    row_len = 4
    for row in range(4):
        for col in range(row_len):
            idx = row * row_len + col
            tri_pts[idx] = [col + 0.5 * row, row * np.sqrt(3) / 2]
    initial_configs.append(tri_pts)

    # --- Optimization Routines ---
    
    bnds = [(-10, 10)] * (2 * (N - 2))
    
    def polish(params, time_limit):
        """Local polish using smooth then exact objective."""
        best_params = params.copy()
        best_score = exact_objective(best_params)
        
        # Phase 1: Smooth L-BFGS-B
        if time.time() < deadline - time_limit - 5:
            try:
                res = minimize(smooth_objective, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < best_score:
                    best_params = res.x
                    best_score = exact_objective(best_params)
            except Exception:
                pass
        
        # Phase 2: Exact L-BFGS-B
        if time.time() < deadline - time_limit - 2:
            try:
                res2 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res2.fun < best_score:
                    best_params = res2.x
                    best_score = res2.fun
            except Exception:
                pass
        
        return best_params, best_score

    best_params = None
    best_score = float('inf')
    
    # 1. Local optimization from diverse starts
    # Allocate ~30s for this phase
    time_per_local = 6.0 
    for cfg in initial_configs:
        if time.time() > deadline - 100: # Leave 100s for DE and final polish
            break
        
        init_params, perm = normalize_points(cfg)
        if init_params is None:
            continue
        
        # Quick check before expensive optimization
        init_score = exact_objective(init_params)
        if init_score < best_score:
            best_score = init_score
            best_params = init_params
        
        # Polish
        params, score = polish(init_params, time_per_local)
        if score < best_score:
            best_score = score
            best_params = params
    
    if best_params is None:
        # Fallback
        init_params, _ = normalize_points(initial_configs[0])
        best_params = init_params
        best_score = exact_objective(best_params)

    # 2. Global Search: Differential Evolution
    # Allocate remaining time minus a safety buffer for final polish
    # DE is stochastic and can be slow, so we cap iterations.
    de_start = time.time()
    de_time_budget = 60.0 # Max 60s for DE
    de_deadline = de_start + de_time_budget
    
    # Check if we have time
    if time.time() < deadline - 20: # Need 20s left for final polish
        try:
            # Use a subset of bounds for DE to keep it efficient? 
            # No, use full bounds but maybe tighter if we have a good solution.
            # If best_score is good, tighten bounds around best_params.
            if best_score < 1.5:
                # Tighter bounds based on current best
                center = best_params
                spread = 5.0
                de_bnds = [(center[i] - spread, center[i] + spread) for i in range(len(center))]
            else:
                de_bnds = bnds
            
            # Differential Evolution
            # maxiter is a soft limit, we also check time inside callback if possible, 
            # but DE doesn't have a clean time-based stop in standard scipy interface easily without wrapping.
            # We'll rely on maxiter and maxfev to approximate time.
            # 14 vars. Population size 10x = 140.
            # Let's limit maxfev to roughly 1000-2000 to stay within time.
            
            de_res = differential_evolution(
                smooth_objective, 
                de_bnds, 
                maxiter=50, 
                maxfev=1500,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False # We will polish the result manually
            )
            
            if de_res.fun < best_score:
                # DE returns smooth objective, we need exact score
                de_exact = exact_objective(de_res.x)
                if de_exact < best_score:
                    best_score = de_exact
                    best_params = de_res.x
        except Exception as e:
            pass # Fail safe

    # 3. Final High-Precision Polish
    if time.time() < deadline - 3:
        # Try a few different polishing strategies on the best candidate found
        candidates = [best_params]
        
        # If we have time, try polishing from a random perturbation of the best solution
        # to escape local minima
        if time.time() < deadline - 10:
            perturbed = best_params + np.random.normal(0, 0.1, size=best_params.shape)
            candidates.append(perturbed)
        
        for cand in candidates:
            if time.time() > deadline - 3:
                break
            try:
                # Final tight polish
                res_final = minimize(exact_objective, cand, method='L-BFGS-B', bounds=bnds,
                                     options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if res_final.fun < best_score:
                    best_params = res_final.x
                    best_score = res_final.fun
            except Exception:
                pass

    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts
