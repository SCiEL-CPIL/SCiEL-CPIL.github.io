"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_pairwise_dists(pts):
        # Compute pairwise distances efficiently
        # pts: (N, D)
        # Returns: (N, N) matrix of distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists

    def score(pts):
        dists = get_pairwise_dists(pts)
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def force_based_step(pts, k, learning_rate):
        """
        Apply force-directed relaxation.
        Force on point i: sum over j != i of:
          if dist < k: repulsion (push apart)
          if dist > k: attraction (pull together)
        This drives all distances towards k.
        """
        dists = get_pairwise_dists(pts)
        forces = np.zeros_like(pts)
        
        # Mask diagonal to avoid self-interaction
        mask = ~np.eye(N, dtype=bool)
        
        # Vectorized force calculation
        # We want forces to push points apart if close, pull together if far.
        # Let ideal distance be k.
        # Force magnitude proportional to (k - dist) ? 
        # If dist < k, (k - dist) > 0 -> Repulsive force (away from neighbor).
        # If dist > k, (k - dist) < 0 -> Attractive force (towards neighbor).
        
        # Unit vectors from i to j: (pts[j] - pts[i]) / dist
        # diff[i,j] = pts[i] - pts[j]
        # We want force on i due to j.
        # If dist < k, i should move away from j. Direction: pts[i] - pts[j].
        # If dist > k, i should move towards j. Direction: pts[j] - pts[i].
        # So force direction is always (pts[i] - pts[j]) * sign(k - dist) / dist?
        # Let's use: F_ij = (pts[i] - pts[j]) * (k - dist_ij) / dist_ij^2
        # This is a standard spring-like force centered at k.
        
        # To avoid division by zero, add epsilon
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        # diff: (N, N, D) where diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        
        # Coefficient for each pair
        # coeff[i,j] = (k - dists[i,j]) / dists[i,j]^2
        coeff = (k - dists) / (dists_safe ** 2)
        # Zero out diagonal
        np.fill_diagonal(coeff, 0)
        
        # Force on i = sum_j coeff[i,j] * diff[i,j]
        # coeff: (N, N), diff: (N, N, D)
        # We need to multiply coeff[i,j] with diff[i,j, :] and sum over j.
        # force[i] = sum_j coeff[i,j] * (pts[i] - pts[j])
        
        # Using einsum or broadcasting
        # coeff_expanded: (N, N, 1)
        forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Update positions
        new_pts = pts + learning_rate * forces
        
        # Re-center to prevent drift
        new_pts -= new_pts.mean(axis=0)
        
        return new_pts

    def smoothed_score(pts):
        """
        A differentiable surrogate for the max/min ratio.
        We want to minimize the variance of distances and maximize the minimum.
        Objective: sum((d_i - mean_d)^2) / mean_d^2  +  penalty if min_d is too small?
        Actually, minimizing the spread of distances is key.
        Let's minimize: (max_d - min_d)^2 / mean_d^2 + 1?
        The raw score is (max/min)^2.
        A good smooth approximation is the sum of squared relative deviations from the median.
        """
        dists = get_pairwise_dists(pts)
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        # If any distance is near zero, return large value
        if d.min() < 1e-8:
            return 1e10
            
        mean_d = d.mean()
        # Variance of distances normalized by mean squared
        variance = np.var(d) / (mean_d ** 2)
        
        # Also penalize if min is much smaller than mean
        # ratio_min = d.min() / mean_d
        # We want ratio_min to be close to 1.
        # Let's combine: variance + (1 - ratio_min)^2 * 10
        ratio_min = d.min() / mean_d
        penalty = (1 - ratio_min) ** 2 * 10
        
        return variance + penalty

    best_pts = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for final polish
            break
            
        # Initialize: Random points on a sphere or in a disk
        # A spherical distribution is often better for 2D packing in terms of distance equalization
        # But 2D Tammes problem solutions are often hexagonal-like or circular.
        # Let's start with a circle perturbed.
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.1, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.column_stack((x, y))
        
        # Normalize scale
        pts /= np.max(np.linalg.norm(pts, axis=1))
        
        # Force-based relaxation
        # Start with a large k (ideal distance) and learning rate
        # We need to estimate a good k.
        # For N points in 2D, average distance is roughly sqrt(N) / sqrt(area).
        # Let's just start with k = 1.0 and let it find its way.
        k = 1.0
        lr = 0.1
        
        for step in range(200):
            if time.time() > deadline - 10:
                break
                
            pts = force_based_step(pts, k, lr)
            
            # Adaptive k: move k towards the current median distance?
            # Or just keep k fixed? 
            # If k is fixed, it forces all distances to k, which is impossible for non-regular graphs.
            # Better: Let k be the current mean distance.
            dists = get_pairwise_dists(pts)
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            k = d.mean()
            
            # Cool down learning rate
            lr *= 0.98
            
            # Re-center
            pts -= pts.mean(axis=0)
            
            # Scale normalization to prevent collapse/explosion
            # Keep max distance around 1?
            # Actually, scale invariance means we can fix max distance to 1.
            # Let's normalize so max distance is 1.
            d_max = d.max()
            if d_max > 0:
                pts /= d_max
        
        # Local refinement with L-BFGS-B on the smoothed score
        # We minimize smoothed_score.
        # Gradient is not provided, so L-BFGS-B will use numerical gradients.
        # This might be slow. Let's try Nelder-Mead or BFGS.
        # BFGS is quasi-Newton and usually faster than NM for smooth functions.
        
        try:
            res = minimize(
                smoothed_score, 
                pts, 
                method='BFGS', 
                options={'maxiter': 100, 'gtol': 1e-4}
            )
            if res.success:
                pts = res.x
                pts -= pts.mean(axis=0)
                
                # Normalize scale to unit max distance for consistent scoring?
                # Score is scale invariant, but let's keep it normalized for stability.
                dists = get_pairwise_dists(pts)
                iu = np.triu_indices(N, 1)
                d = dists[iu]
                d_max = d.max()
                if d_max > 0:
                    pts /= d_max
                    
                curr_score = score(pts)
                
                if curr_score < best_score:
                    best_score = curr_score
                    best_pts = pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback: Grid
        side = int(np.ceil(N ** 0.5))
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append([i, j])
                if len(grid) == N:
                    break
            if len(grid) == N:
                break
        best_pts = np.array(grid, dtype=float)
        best_pts -= best_pts.mean(axis=0)

    return best_pts
# EVOLVE-BLOCK-END
