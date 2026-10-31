import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_pairwise_dists(pts):
        """Calculate pairwise distances."""
        # pts is (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return np.sqrt(dists_sq[iu])

    def exact_score(pts):
        """Calculate the exact score: (max dist / min dist)^2"""
        if len(pts) < 2:
            return 1e10
        dists = get_pairwise_dists(pts)
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    def soft_min_max(dists, beta):
        """
        Calculate soft min and max of distances.
        Uses log-sum-exp for stability.
        """
        # Soft Max
        # max(d) ~ (1/beta) * log(sum(exp(beta * d)))
        # To avoid overflow, subtract max(d)
        d_max = np.max(dists)
        exp_terms_max = np.exp(beta * (dists - d_max))
        sum_exp_max = np.sum(exp_terms_max)
        if sum_exp_max < 1e-300:
            soft_max = d_max
        else:
            soft_max = d_max + (1.0 / beta) * np.log(sum_exp_max)
        
        # Soft Min
        # min(d) ~ -(1/beta) * log(sum(exp(-beta * d)))
        d_min = np.min(dists)
        exp_terms_min = np.exp(-beta * (dists - d_min))
        sum_exp_min = np.sum(exp_terms_min)
        if sum_exp_min < 1e-300:
            soft_min = d_min
        else:
            soft_min = d_min - (1.0 / beta) * np.log(sum_exp_min)
            
        return soft_max, soft_min

    def soft_objective(params, beta):
        """
        Objective function for optimization.
        params: flattened coordinates of N points (2N values).
        beta: sharpness parameter.
        """
        pts = params.reshape(N, 2)
        
        # Check for duplicates (optional but good for robustness)
        dists = get_pairwise_dists(pts)
        if np.min(dists) < 1e-6:
            return 1e10 + (1e-6 - np.min(dists))**2 * 1e6

        soft_max, soft_min = soft_min_max(dists, beta)
        
        if soft_min < 1e-8:
            return 1e10
            
        ratio = soft_max / soft_min
        return ratio ** 2

    def exact_objective(params):
        """Exact objective for final polishing."""
        pts = params.reshape(N, 2)
        return exact_score(pts)

    def normalize(pts):
        """
        Normalize points to reduce scale invariance issues during optimization.
        Center at origin and scale such that min distance is 1? 
        Or just center and scale to unit RMS?
        For this problem, scale invariance means we can fix one point or scale.
        However, L-BFGS-B works better if variables are O(1).
        Let's center at origin and scale such that max distance is 1.
        """
        pts = pts - np.mean(pts, axis=0)
        dists = get_pairwise_dists(pts)
        max_d = np.max(dists)
        if max_d > 1e-8:
            pts = pts / max_d
        return pts

    # Initial configurations
    
    # 1. Hexagonal Lattice
    # Generate a larger hex lattice and pick the 16 closest to center
    R = 5
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Distance from origin
    dists_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_origin)
    init_hex = hex_pts[sorted_indices[:N]]
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 3. 4x4 Grid
    grid_x = np.linspace(-1.5, 1.5, 4)
    grid_y = np.linspace(-1.5, 1.5, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    init_grid = np.column_stack([gx.flatten(), gy.flatten()])
    
    # 4. Golden Angle Spiral
    # This creates a well-distributed set of points
    init_spiral = np.zeros((N, 2))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(N):
        r = np.sqrt(i + 0.5)
        theta = i * golden_angle
        init_spiral[i, 0] = r * np.cos(theta)
        init_spiral[i, 1] = r * np.sin(theta)
        
    # 5. Random
    np.random.seed(42)
    init_random = np.random.uniform(-2, 2, (N, 2))
    
    # 6. 5x5 grid minus corners? No, 16 points. 4x4 is better.
    # Try a "flower" pattern: 1 center + 6 inner ring + 9 outer?
    # Let's stick to the above.

    candidates = [init_hex, init_circle, init_grid, init_spiral, init_random]
    
    # Add some perturbed versions of the best candidates
    # We will run optimization on each
    
    best_score = 1e10
    best_params = None
    
    bnds = [(-5, 5)] * (2 * N)
    
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        # Normalize initial points
        init_pts_norm = normalize(init_pts)
        init_params = init_pts_norm.flatten()
        
        # Optimization phases with annealing
        # Phase 1: Coarse (low beta)
        betas = [5.0, 20.0, 100.0, 500.0]
        current_params = init_params.copy()
        
        for i, beta in enumerate(betas):
            if time.time() > deadline - 10:
                break
                
            try:
                # Use L-BFGS-B
                # maxiter depends on time remaining
                maxiter = 200 if i == 0 else 100
                res = minimize(
                    soft_objective, 
                    current_params, 
                    method='L-BFGS-B', 
                    bounds=bnds, 
                    args=(beta,),
                    options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.fun < 1e9:
                    current_params = res.x
            except Exception:
                pass
        
        # Final Polish with exact objective
        if time.time() < deadline - 5:
            try:
                res_exact = minimize(
                    exact_objective, 
                    current_params, 
                    method='L-BFGS-B', 
                    bounds=bnds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_exact.fun < 1e9:
                    current_params = res_exact.x
            except Exception:
                pass
            
            # Nelder-Mead for final local refinement
            try:
                res_nm = minimize(
                    exact_objective, 
                    current_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if res_nm.fun < 1e9:
                    current_params = res_nm.x
            except Exception:
                pass
        
        # Evaluate final score
        final_score = exact_objective(current_params)
        
        if final_score < best_score:
            best_score = final_score
            best_params = current_params.copy()

    # If no valid params found, fallback to hex
    if best_params is None:
        best_params = normalize(init_hex).flatten()
        
    final_pts = best_params.reshape(N, 2)
    
    # Ensure distinct points
    # If any points are too close, perturb them slightly
    dists = get_pairwise_dists(final_pts)
    if np.min(dists) < 1e-5:
        # Simple perturb
        final_pts += np.random.uniform(-1e-3, 1e-3, final_pts.shape)
        
    return final_pts

# EVOLVE-BLOCK-END
