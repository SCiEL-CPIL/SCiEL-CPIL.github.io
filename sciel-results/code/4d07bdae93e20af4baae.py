import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    # 1. Generate initial candidate points from a hexagonal lattice
    # We want to find a subset of 16 points that is "spread out" but compact.
    # A hexagonal lattice is optimal for packing, but we need to select 16.
    # A good heuristic is to take a hexagonal number region.
    # 16 is not a hexagonal number (1, 7, 19...).
    # We can take the central 19 and remove 3, or take a distorted shape.
    
    R = 6
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    pts = np.array(pts)
    
    # Calculate distances from origin
    dists_origin = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_origin)
    
    # Strategy: Try several different initial subsets
    # 1. Closest 16 to origin
    init_sets = []
    init_sets.append(sorted_idx[:16])
    
    # 2. Closest 19 to origin, remove 3 most "central" or redundant? 
    # Actually, just taking the next few and optimizing is often enough.
    # Let's also try a "ring" based selection to encourage larger spread.
    # But for now, the continuous optimizer is strong enough if given a good start.
    
    best_params = None
    best_score = float('inf')

    # Helper to convert indices to params (fixed first two points)
    def indices_to_params(indices, all_pts):
        sub_pts = all_pts[indices]
        # Find min distance pair to fix
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = sub_pts[i0]
        p1 = sub_pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return None
            
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (sub_pts - p0) @ R_mat.T / norm_v
        
        # Reorder
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed[2:].flatten()

    # Helper to evaluate exact score
    def exact_score(params):
        if params is None:
            return float('inf')
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10
        return max_d_sq / min_d_sq

    # Smooth objective with log-sum-exp
    def soft_objective(params, beta):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        max_d = np.max(d_sq)
        min_d = np.min(d_sq)
        
        # Soft max: log-sum-exp(beta * x) / beta
        shifted_max = d_sq - max_d
        soft_max = max_d + np.log(np.sum(np.exp(beta * shifted_max))) / beta
        
        # Soft min: -log-sum-exp(-beta * x) / beta
        shifted_min = d_sq - min_d
        soft_min = min_d - np.log(np.sum(np.exp(-beta * shifted_min))) / beta
        
        if soft_min < 1e-10:
            return 1e10
        return soft_max / soft_min

    # Exact gradient for L-BFGS-B
    def exact_objective_grad(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10, np.zeros_like(params)
        
        obj = max_d_sq / min_d_sq
        
        min_idx = np.argmin(d_sq)
        max_idx = np.argmax(d_sq)
        
        min_pair_i, min_pair_j = iu[0][min_idx], iu[1][min_idx]
        max_pair_i, max_pair_j = iu[0][max_idx], iu[1][max_idx]
        
        grad_pts = np.zeros_like(pts_opt)
        
        # d(obj)/d(d_sq_ij)
        # obj = M / m
        # d(obj)/d(m) = -M/m^2
        # d(obj)/d(M) = 1/m
        
        v_min = pts_opt[min_pair_i] - pts_opt[min_pair_j]
        coeff_min = -max_d_sq / (min_d_sq**2)
        grad_pts[min_pair_i] += 2 * v_min * coeff_min
        grad_pts[min_pair_j] += -2 * v_min * coeff_min
        
        v_max = pts_opt[max_pair_i] - pts_opt[max_pair_j]
        coeff_max = 1.0 / min_d_sq
        grad_pts[max_pair_i] += 2 * v_max * coeff_max
        grad_pts[max_pair_j] += -2 * v_max * coeff_max
        
        grad_params = grad_pts[2:].flatten()
        
        return obj, grad_params

    bnds = [(-20, 20)] * (2 * (N - 2))

    # Try the initial lattice subset
    for init_indices in init_sets:
        if time_left() < 10:
            break
            
        init_params = indices_to_params(init_indices, pts)
        if init_params is None:
            continue
        
        current_score = exact_score(init_params)
        if current_score < best_score:
            best_score = current_score
            best_params = init_params.copy()
            
        # Optimization Phases
        current_params = init_params.copy()
        
        # Phase 1: Smooth surrogate optimization
        # Start with low beta, increase
        for beta in [10.0, 20.0, 50.0, 100.0]:
            if time_left() < 15:
                break
            try:
                res = minimize(lambda p: soft_objective(p, beta), current_params, 
                               method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e9:
                    # Check exact score
                    exact_s = exact_score(res.x)
                    if exact_s < best_score:
                        best_score = exact_s
                        best_params = res.x.copy()
                        current_params = res.x.copy()
                    else:
                        # Even if not better globally, it might be a better local start
                        # for the next beta or exact phase
                        current_params = res.x.copy()
            except Exception:
                pass
        
        # Phase 2: Exact optimization with gradient
        if time_left() < 10:
            break
        try:
            res = minimize(exact_objective_grad, current_params, 
                           method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                current_params = res.x.copy()
        except Exception:
            pass

        # Phase 3: High precision polish
        if time_left() < 5:
            break
        try:
            res = minimize(exact_objective_grad, current_params, 
                           method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If we still have time, try a few random restarts from the best solution
    if time_left() > 20:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time_left() < 5:
                break
            scale = 0.1
            perturbation = rng.normal(0, scale, size=best_params.shape)
            restart_params = best_params + perturbation
            try:
                res = minimize(exact_objective_grad, restart_params, 
                               method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Construct final points
    if best_params is None:
        # Fallback: return the initial lattice subset
        final_pts = pts[init_sets[0]]
    else:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Validate distinctness
        diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        min_dist = np.min(dists[iu])
        
        if min_dist < 1e-8:
            # Fallback if points collapsed
            final_pts = pts[init_sets[0]]

    return final_pts
