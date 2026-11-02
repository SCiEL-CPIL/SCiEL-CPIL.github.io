"""Improved packing of 50 equal circles in a unit square via force-directed relaxation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    if len(centres) < 2:
        min_pair = np.inf
    else:
        min_pair = pdist(centres, metric='euclidean').min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _construct_grid():
    """Create a grid-based initial configuration."""
    # 8x7 grid = 56 points, take first 50
    ticks_x = np.linspace(-0.5 + 1/16, 0.5 - 1/14, 8)
    ticks_y = np.linspace(-0.5 + 1/14, 0.5 - 1/14, 7)
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N]

def _relax(centres, iterations=5000, step_size=0.01):
    """
    Relax the configuration by minimizing the sum of inverse squared distances.
    This tends to maximize the minimum pairwise distance.
    """
    centres = centres.copy()
    for i in range(iterations):
        # Decay step size slowly
        lr = step_size * (1.0 - i / iterations)
        if lr <= 0:
            break
            
        forces = np.zeros_like(centres)
        
        # Calculate pairwise forces
        # d[i, j] is distance between i and j
        # We want to minimize sum(1/d^2)
        # Gradient w.r.t c_i is sum_j -2/d^3 * (c_i - c_j)
        
        # Efficient calculation using broadcasting
        # diff[i, j, k] = c_i[k] - c_j[k]
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-10
        
        # We only care about pairs i < j to avoid double counting in the sum,
        # but for forces, we can compute full matrix and sum.
        # Force on i from j: -2 / dist^3 * (c_i - c_j)
        # This is equivalent to: 2 / dist^3 * (c_j - c_i)
        
        # dist^3 = dist_sq^(3/2)
        dist_cubed = dist_sq ** 1.5
        
        # Coefficient for each pair
        # coeff[i, j] = 2 / dist_cubed[i, j]
        coeff = 2.0 / dist_cubed
        
        # Zero out diagonal (self-interaction)
        np.fill_diagonal(coeff, 0.0)
        
        # Force on i is sum_j coeff[i, j] * (c_j - c_i)
        # = sum_j coeff[i, j] * c_j - sum_j coeff[i, j] * c_i
        # = (coeff.T @ centres) - (coeff.sum(axis=1)[:, np.newaxis] * centres)
        
        # Wait, coeff[i,j] is symmetric.
        # Force_i = sum_j coeff[i,j] * (c_j - c_i)
        # Let's compute term1 = sum_j coeff[i,j] * c_j = coeff @ centres
        # term2 = sum_j coeff[i,j] * c_i = coeff.sum(axis=1)[:, np.newaxis] * centres
        
        term1 = coeff @ centres
        term2 = coeff.sum(axis=1)[:, np.newaxis] * centres
        
        forces = term1 - term2
        
        # Update positions
        centres += lr * forces
        
        # Project to square [-0.5, 0.5]
        centres = np.clip(centres, -0.5, 0.5)
        
    return centres

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    best_centres = None
    best_score = -1.0
    
    # Multiple restarts with different initial configurations
    for seed in range(10):
        np.random.seed(seed)
        
        # Start with a grid, then add noise
        base = _construct_grid()
        noise = np.random.normal(0, 0.05, size=(N, 2))
        centres = base + noise
        centres = np.clip(centres, -0.5, 0.5)
        
        # Relax
        relaxed = _relax(centres, iterations=3000, step_size=0.05)
        
        # Further refinement with smaller steps
        refined = _relax(relaxed, iterations=2000, step_size=0.01)
        
        score = _radius_score(refined)
        
        if score > best_score:
            best_score = score
            best_centres = refined
    
    return best_centres
# EVOLVE-BLOCK-END
