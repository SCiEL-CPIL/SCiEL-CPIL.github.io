"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import pdist

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    def prepare_params(pts):
        """
        Reduces dimensionality by fixing the first two points (closest pair)
        to (0,0) and (1,0).
        """
        d = pdist(pts, 'euclidean')
        min_idx = np.argmin(d)
        iu = np.triu_indices(N, 1)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
             return None, None

        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i_min is 0 and j_min is 1
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # transformed[0] should be [0,0]
        # transformed[1] should be [1,0]
        params = transformed[2:].flatten()
        return params, (i_min, j_min)

    def objective(params):
        """
        Objective function for optimization.
        Reconstructs points from params (p0=0, p1=1, rest=params) and computes score.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        return score_points(pts)

    def generate_seeds():
        """Generate various initial configurations."""
        seeds = []
        
        # 1. Square Grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        seeds.append(grid)
        
        # 2. Hexagonal Lattice subset
        # Center at 0,0
        hex_pts = []
        R = 4
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        # Take closest 16 to origin
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        seeds.append(hex_pts[idx])
        
        # 3. Circle Packing (Fibonacci/Lattice on circle)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        seeds.append(circle_pts)
        
        # 4. Random Uniform
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-1, 1, size=(16, 2))
        seeds.append(rand_pts)
        
        # 5. Two concentric rings
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.5
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(ring_pts)
        
        # 6. Hexagonal Lattice with swap optimization (Inspiration 1)
        # Generate a larger pool of hexagonal points
        R_pool = 8
        pool_pts = []
        for q in range(-R_pool, R_pool + 1):
            for r in range(-R_pool, R_pool + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pool_pts.append((x, y))
        pool_pts = np.array(pool_pts)
        
        # Sort by distance from origin to get a good initial subset
        dists_from_origin = np.sum(pool_pts**2, axis=1)
        sorted_indices = np.argsort(dists_from_origin)
        
        # Initial subset: 16 closest
        current_indices = sorted_indices[:N].copy()
        
        # Define a helper to score a subset of indices from the pool
        def score_subset(indices):
            sub_pts = pool_pts[indices]
            diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            min_d = np.min(d)
            max_d = np.max(d)
            if min_d < 1e-8:
                return 1e10
            return (max_d / min_d)**2

        current_score = score_subset(current_indices)
        in_subset = set(current_indices)
        
        # Candidate pool: top 100 closest points to origin
        pool_size = 100
        candidate_indices = sorted_indices[:pool_size]
        outside_candidates = [i for i in candidate_indices if i not in in_subset]
        
        # Greedy swap optimization
        # Limit iterations to avoid taking too long
        max_swap_iters = 50
        for _ in range(max_swap_iters):
            if time.time() > deadline - 20:
                break
            improved = False
            for i_idx, i in enumerate(current_indices):
                if time.time() > deadline - 20:
                    break
                for j in outside_candidates:
                    if time.time() > deadline - 20:
                        break
                    new_indices = current_indices.copy()
                    new_indices[i_idx] = j
                    new_score = score_subset(new_indices)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        current_indices = new_indices
                        in_subset = set(current_indices)
                        outside_candidates = [k for k in candidate_indices if k not in in_subset]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        best_hex_swap_pts = pool_pts[current_indices]
        seeds.append(best_hex_swap_pts)

        return seeds

    best_score = 1e10
    best_params = None

    seeds = generate_seeds()
    
    for seed_idx, seed_pts in enumerate(seeds):
        if time.time() > deadline - 15:
            break
            
        params, info = prepare_params(seed_pts)
        if params is None:
            continue
            
        # Initial score
        curr_score = objective(params)
        if curr_score < best_score:
            best_score = curr_score
            best_params = params.copy()
            
        # Optimize with Powell
        try:
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 500, 'maxfev': 2000, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception as e:
            pass

    if best_params is None:
        # Fallback
        return np.random.rand(N, 2)

    # Refinement Phase 1: Soft Objective with L-BFGS-B
    # Use log-sum-exp for smooth max/min
    def soft_objective(params, beta=100.0):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max of d_sq
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min of d_sq
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        # We want to minimize (max/min)^2, which is equivalent to minimizing log(max^2/min^2) = 2(log(max) - log(min))
        # Or just (soft_max_d_sq / soft_min_d_sq)
        return soft_max_d_sq / soft_min_d_sq

    # Increase beta gradually for better conditioning? Or just use a high beta.
    # L-BFGS-B with finite difference gradients can be slow if beta is too high (sharp peaks).
    # Let's try beta=50 first.
    
    if time.time() < deadline - 10:
        try:
            res2 = minimize(soft_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            exact_score = objective(res2.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res2.x.copy()
        except Exception:
            pass

    # Refinement Phase 2: Epigraph Formulation with Auxiliary Variables
    # Minimize t
    # Subject to:
    #   d_ij^2 <= t * d_min^2  (for all pairs)  => d_ij^2 / d_min^2 <= t
    #   d_min^2 >= epsilon
    # This is still non-convex due to d_min^2 in denominator.
    
    # Alternative:
    # Let u = 1/d_min. Minimize t * u^2 ? No.
    # Minimize R^2 where R = d_max / d_min.
    # d_max <= R * d_min.
    # d_ij <= R * d_min for all i,j.
    # d_min = min(d_ij).
    
    # Let's try a different parametrization or penalty.
    # Actually, the soft objective usually works well.
    # Let's try a final polish with Nelder-Mead on the exact objective if time permits.
    
    if time.time() < deadline - 5:
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            
            exact_score = objective(res3.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res3.x.copy()
        except Exception:
            pass

    # Reconstruct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify
    final_score = score_points(final_pts)
    if final_score < best_score:
        best_score = final_score

    return final_pts

# EVOLVE-BLOCK-END
