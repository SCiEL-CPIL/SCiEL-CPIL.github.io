"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
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

    def soft_objective(params, beta=50.0):
        """
        Smooth surrogate for the objective function to allow gradient-based optimization
        of the min/max terms.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max
        max_d_sq = np.max(d_sq)
        # Use log-sum-exp trick for stability
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def generate_hex_seed():
        """
        Generate a hexagonal lattice subset of 16 points.
        This is a strong geometric prior for 2D packing.
        """
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        
        # Select 16 points closest to the origin to form a compact cluster
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return hex_pts[idx]

    def generate_other_seeds():
        """Generate alternative initial configurations."""
        seeds = []
        
        # Square Grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        seeds.append(grid)
        
        # Two concentric rings
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.5
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(ring_pts)
        
        # Random Uniform
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-1, 1, size=(16, 2))
        seeds.append(rand_pts)
        
        return seeds

    best_score = 1e10
    best_params = None

    # 1. Primary Seed: Hexagonal Lattice
    # This is often the global optimum or very close to it for small N
    hex_seed = generate_hex_seed()
    
    # 2. Secondary Seeds
    other_seeds = generate_other_seeds()
    
    all_seeds = [hex_seed] + other_seeds
    
    # Add perturbed versions of the hex seed to escape potential local minima
    rng = np.random.default_rng(123)
    for _ in range(3):
        perturbed = hex_seed + rng.normal(0, 0.05, size=(16, 2))
        all_seeds.append(perturbed)

    for seed_pts in all_seeds:
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
            
        # Optimization Phase 1: Powell (Derivative-free, good for non-smooth start)
        try:
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 200, 'maxfev': 2000, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception as e:
            pass

        # Optimization Phase 2: L-BFGS-B on Soft Objective (Gradient-based polish)
        # This helps refine the solution to high precision
        if time.time() < deadline - 10:
            try:
                # Increase beta for sharper approximation to exact min/max
                res2 = minimize(soft_objective, res.x, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                
                # Check exact score of the result
                exact_score = objective(res2.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_params = res2.x.copy()
                    
            except Exception:
                pass

    # Final Polish: If we have time, do one more very tight L-BFGS-B pass on the best found
    if best_params is not None and time.time() < deadline - 5:
        try:
            # Use a very high beta for near-exact gradient
            def final_soft_objective(params, beta=200.0):
                return soft_objective(params, beta)
                
            res3 = minimize(final_soft_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False})
            
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
    
    return final_pts

# EVOLVE-BLOCK-END
