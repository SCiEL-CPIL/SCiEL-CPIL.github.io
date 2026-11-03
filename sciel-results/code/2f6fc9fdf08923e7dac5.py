"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _compute_score(centers):
    """
    Compute the score for a given set of centers.
    Score = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    Returns -1 if invalid (centers outside cube, non-finite, etc.)
    """
    if not np.all(np.isfinite(centers)):
        return -1.0
    
    # Check if centers are within the cube [-0.5, 0.5]^3
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return -1.0
    
    # Clamp centers to be strictly inside for safety in score calculation
    # Actually, the problem says "inside the cube [-1/2, 1/2]^3"
    # A centre on the boundary is allowed? "A centre on or outside the cube... make the candidate invalid"
    # So centers must be strictly inside? Or on the boundary is invalid?
    # "A centre on or outside the cube, coincident centres, a wrong shape or non-finite values make the candidate invalid."
    # This implies centers must be strictly inside (-0.5, 0.5).
    # However, the score formula uses (1/2 - |c_ik|). If c_ik = 0.5, score is 0.
    # But the validity check says "on or outside" is invalid.
    # So we must ensure centers are strictly inside.
    # Let's add a small margin to ensure validity during optimization, then clamp at the end.
    # Actually, for the optimizer, we can work with a slightly smaller box to ensure validity,
    # or just penalize.
    
    # Calculate pairwise distances
    # Use broadcasting for efficiency
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # More efficient:
    sq_dists = np.sum(centers**2, axis=1)
    dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * centers @ centers.T
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # min_i min_k (1/2 - |c_ik|)
    # This is equivalent to 0.5 - max_i max_k |c_ik|
    max_abs_coord = np.max(np.abs(centers))
    min_wall_dist = 0.5 - max_abs_coord
    
    score = min(min_pair_dist / 2.0, min_wall_dist)
    
    # Check validity: centers must be strictly inside
    if np.any(centers <= -0.5) or np.any(centers >= 0.5):
        return -1.0
    
    return score


def _neg_score(centers_flat):
    """Negative score for minimization."""
    centers = centers_flat.reshape(-1, 3)
    score = _compute_score(centers)
    if score < 0:
        return 1.0  # Large penalty for invalid
    return -score


def construct():
    # 1. Initial Guess: Simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N]
    
    # Ensure initial centers are strictly inside
    initial_centers = np.clip(initial_centers, -0.49, 0.49)
    
    # 2. Local Optimization: Use L-BFGS-B or similar to maximize the minimum clearance
    # We will maximize r such that:
    # |c_i - c_j| >= 2r for all i<j
    # |c_ik| <= 0.5 - r for all i,k
    
    # This is a non-convex problem. We can use a heuristic local search.
    # Let's try a few iterations of gradient-free local search or use scipy's differential_evolution
    # for a global-ish search, then refine.
    
    # Given 120s limit, differential_evolution on 150 variables might be too slow.
    # Let's try a simpler approach: 
    # 1. Start with a good initial guess (grid).
    # 2. Use a local optimizer that maximizes the minimum distance.
    
    # We can formulate this as: maximize r
    # subject to |c_i - c_j|^2 - (2r)^2 >= 0
    #            (0.5 - r)^2 - c_ik^2 >= 0
    
    # Let's use a simple iterative algorithm:
    # - Compute current r
    # - Find the most "constrained" pairs and walls
    # - Move centers to alleviate constraints
    
    # Alternatively, use scipy.optimize.minimize with method='Nelder-Mead' on the negative score.
    # But Nelder-Mead is slow for 150 dims.
    
    # Let's try a hybrid:
    # 1. Use a random restart local search.
    # 2. For each restart, use a simple gradient ascent on the log-sum-exp approximation of the min distance?
    #    Or just use the exact min distance with subgradient.
    
    # Let's try a simpler approach first: 
    # Use a grid of initial positions and pick the best one after a short local optimization.
    
    # Actually, the known best packing is 0.13595. The grid gives:
    # m=4, grid spacing 0.25.
    # min pair dist = 0.25. r = 0.125.
    # min wall dist = 0.5 - 0.375 = 0.125.
    # So grid score is 0.125.
    
    # We need to get to ~0.136.
    
    # Let's try to use a local search that pushes spheres apart.
    
    def _local_search(centers, max_iters=1000, step=0.01):
        current_score = _compute_score(centers)
        if current_score < 0:
            return centers, 0.0
        
        for _ in range(max_iters):
            # Compute gradients (subgradients)
            # grad_i = sum_{j != i} (c_i - c_j) / |c_i - c_j|  (for pair constraints)
            #         + sign(c_ik) for wall constraints if active
            
            # This is complex. Let's use a simple random perturbation + hill climbing.
            improved = False
            for _ in range(10):
                perturbed = centers + np.random.normal(0, step, size=centers.shape)
                # Keep inside cube
                perturbed = np.clip(perturbed, -0.49, 0.49)
                new_score = _compute_score(perturbed)
                if new_score > current_score:
                    centers = perturbed
                    current_score = new_score
                    improved = True
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
        
        return centers, current_score
    
    # Run local search from the grid
    best_centers, best_score = _local_search(initial_centers, max_iters=500, step=0.05)
    
    # Try a few random restarts
    for _ in range(5):
        random_centers = np.random.uniform(-0.4, 0.4, size=(N, 3))
        rc, rs = _local_search(random_centers, max_iters=200, step=0.05)
        if rs > best_score:
            best_centers = rc
            best_score = rs
    
    # Final refinement
    best_centers, best_score = _local_search(best_centers, max_iters=2000, step=0.01)
    
    # Ensure strict validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers

# EVOLVE-BLOCK-END
