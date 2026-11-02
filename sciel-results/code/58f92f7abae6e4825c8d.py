# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 300

def _hexagonal_seed(n):
    """
    Generate a high-quality hexagonal lattice seed.
    We place points in a hexagonal grid and select the 'n' closest to the origin
    that fit within a radius slightly larger than the estimated packing radius.
    """
    # Estimate radius
    # Area packing fraction ~ 0.9069 for hexagonal
    r_est = 1.0 / np.sqrt(n * (1.0 / 0.9069))
    spacing = 2.0 * r_est
    
    # Generate grid
    # Hexagonal lattice vectors:
    # v1 = (spacing, 0)
    # v2 = (spacing/2, spacing*sqrt(3)/2)
    
    max_extent = 1.2
    k_max = int(max_extent / spacing) + 2
    
    points = []
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    
    # We want points that are inside the container.
    # The container radius for centers is 1 - r_est.
    # We select the n points closest to the origin that are within 1 - r_est.
    # However, if we just take the closest, we might cut off the hexagonal symmetry.
    # A better way: take all points within 1 - r_est. If count > n, take the n closest.
    # If count < n, we have a problem. We need to adjust spacing.
    
    # Let's iterate on spacing to get exactly n points inside.
    # Since we want to maximize r, we want a dense packing.
    # Let's find the spacing such that we get roughly n points inside 1.0.
    
    # Approximate number of points in circle of radius R with spacing s:
    # Area ~ pi * R^2. Area per point ~ s^2 * sqrt(3)/2.
    # n ~ pi * R^2 / (s^2 * sqrt(3)/2)
    # s ~ sqrt(2 * pi * R^2 / (n * sqrt(3)))
    
    R_target = 1.0
    s_est = np.sqrt(2 * np.pi * R_target**2 / (n * np.sqrt(3)))
    
    for attempt in range(10):
        spacing = s_est * (1.0 + 0.01 * attempt)
        max_extent = 1.2
        k_max = int(max_extent / spacing) + 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append([x, y])
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        mask = dists < 1.0
        points_in = points[mask]
        dists_in = dists[mask]
        
        if len(points_in) >= n:
            # Sort by distance to origin, take closest n
            idx = np.argsort(dists_in)
            return points_in[idx[:n]]
        else:
            # Need smaller spacing
            s_est *= 0.95
            
    # Fallback
    idx = np.argsort(dists)
    return points[idx[:n]]

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    if len(centers) < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(centers, axis=1)
    min_boundary = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_boundary)

def _energy_and_grad(x, r_target):
    """
    Compute energy and gradient for the packing problem.
    Energy = Sum of squared overlaps (pairwise) + Sum of squared boundary violations.
    We use soft penalties to make it differentiable.
    """
    n = len(x) // 2
    centers = x.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    pair_diffs = diff[mask] # Shape (M, 2)
    
    # Overlap amount: max(0, 2r - d)
    overlaps = np.maximum(0, 2 * r_target - pair_dists)
    
    # Energy: Sum of overlaps^2
    e_pair = np.sum(overlaps ** 2)
    
    # Gradient of e_pair
    # d/dx sum max(0, 2r-d)^2
    # Let f(d) = max(0, 2r-d)^2. f'(d) = -2(2r-d) if d < 2r, else 0.
    # d/dx_i f(d_ij) = f'(d_ij) * d d_ij / d x_i
    # d d_ij / d x_i = (x_i - x_j) / d_ij
    # So grad_i = sum_j -2(2r-d_ij) * (x_i - x_j) / d_ij
    # Note: We are summing over unique pairs.
    
    # We need to accumulate gradients for all i and j.
    grad = np.zeros_like(x)
    
    if len(overlaps) > 0:
        # Coefficient for each pair
        coeff = -2.0 * overlaps # Shape (M,)
        # Unit vectors
        # d_ij = |x_i - x_j|. x_i - x_j is pair_diffs.
        # Wait, diff = x_i - x_j. So pair_diffs is x_i - x_j for i<j.
        # d/dx_i (x_i - x_j)/d = (x_i - x_j)/d
        # d/dx_j (x_i - x_j)/d = -(x_i - x_j)/d
        
        unit_vecs = pair_diffs / (pair_dists[:, np.newaxis] + 1e-12) # Shape (M, 2)
        
        # Grad contribution to x_i: coeff * unit_vec
        # Grad contribution to x_j: -coeff * unit_vec
        
        # We need to map pair indices back to i, j
        # This is tricky with flat arrays. Let's use a loop or matrix operations.
        # For N=300, M ~ 45000. A loop in Python is slow.
        # Let's use matrix operations.
        
        # Create full gradient matrix G (n, n) where G[i,j] is the vector force from j on i?
        # No, let's construct the gradient matrix directly.
        
        # Let W be an (n, n) matrix where W[i,j] = coeff_k if (i,j) is pair k, else 0.
        # Then Grad_i = sum_j W[i,j] * (x_i - x_j) / d_ij ?
        # Actually, force on i from j is proportional to (x_i - x_j).
        # Let F_i = sum_{j != i} F_{ij}
        # F_{ij} = -2(2r-d) * (x_i - x_j)/d if d < 2r else 0.
        
        # We can compute this using the full distance matrix.
        
        # Recompute full dists and diffs for clarity and vectorization
        # diff_full = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists_full = sqrt(sum(diff_full^2))
        # overlap_full = max(0, 2r - dists_full)
        # force_mat = overlap_full * (2r - dists_full) ... wait.
        
        # Let's stick to the vectorized full matrix approach, it's O(N^2) memory but N=300 is small (90k floats).
        
        diff_full = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_full = np.sqrt(np.sum(diff_full ** 2, axis=-1) + 1e-12)
        overlaps_full = np.maximum(0, 2 * r_target - dists_full)
        
        # Force magnitude on i from j:
        # F_{ij} = 2 * overlap_{ij} * (2r - d_{ij}) ? No.
        # E = sum_{i<j} overlap_{ij}^2
        # dE/dx_i = sum_{j != i} 2 * overlap_{ij} * d(overlap_{ij})/d(x_i)
        # d(overlap_{ij})/d(x_i) = - (x_i - x_j)/d_{ij} if d < 2r else 0
        # So term is 2 * overlap_{ij} * (-(x_i - x_j)/d_{ij})
        
        # Let's define a weight matrix W where W[i,j] = 2 * overlap_{ij} / d_{ij} if d < 2r else 0.
        # Then Grad_i = - sum_j W[i,j] * (x_i - x_j)
        
        W = np.zeros((n, n))
        # Only calculate where overlap > 0
        mask_overlap = overlaps_full > 0
        # Avoid division by zero
        dists_safe = np.where(dists_full < 1e-12, 1e-12, dists_full)
        W[mask_overlap] = 2.0 * overlaps_full[mask_overlap] / dists_safe[mask_overlap]
        
        # Grad_i = - sum_j W[i,j] * (x_i - x_j)
        # = - (sum_j W[i,j] * x_i - sum_j W[i,j] * x_j)
        # = - x_i * sum_j W[i,j] + sum_j W[i,j] * x_j
        
        sum_W = np.sum(W, axis=1)
        grad_pair = -centers * sum_W[:, np.newaxis] + (W @ centers)
        
        # Boundary Energy
        radii = np.linalg.norm(centers, axis=1)
        bound_overlaps = np.maximum(0, r_target + radii - 1.0)
        e_bound = np.sum(bound_overlaps ** 2)
        
        # Grad bound:
        # E_b = sum max(0, r + |x| - 1)^2
        # dE/dx = 2 * bound_overlap * (x/|x|)
        
        safe_radii = np.where(radii < 1e-12, 1e-12, radii)
        grad_bound = 2.0 * bound_overlaps[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
        
        total_grad = (grad_pair + grad_bound).flatten()
        total_energy = e_pair + e_bound
        
        return total_energy, total_grad

def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Strategy:
    1. Generate hexagonal seed.
    2. Estimate initial radius.
    3. Iteratively increase radius and minimize energy using L-BFGS.
    """
    centers = _hexagonal_seed(N)
    
    # Initial radius estimate based on seed
    # The seed spacing is roughly 2*r.
    # We can compute the actual min distance in the seed to get a lower bound.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    r_seed = np.min(dists) / 2.0
    
    # We want to go higher than r_seed.
    # Let's try to reach r_target = 0.05315 (known best) or higher.
    # Start from r_seed * 0.9 and ramp up.
    
    r_start = r_seed * 0.8
    r_target = 0.0535 # Slightly above known best to try to beat it
    
    # Step 1: Coarse Relaxation with L-BFGS at lower r to settle the lattice
    x0 = centers.flatten()
    
    def obj_coarse(x):
        e, g = _energy_and_grad(x, r_start)
        return e
    
    def grad_coarse(x):
        e, g = _energy_and_grad(x, r_start)
        return g
        
    res1 = minimize(
        obj_coarse, x0, jac=grad_coarse, method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    centers = res1.x.reshape(-1, 2)
    
    # Step 2: Ramp up radius
    # We will perform a series of optimizations, increasing r_target slightly each time.
    # This helps escape local minima where the lattice is too compressed.
    
    n_steps = 20
    r_values = np.linspace(r_start, r_target, n_steps)
    
    for i in range(n_steps):
        r_curr = r_values[i]
        
        def obj_curr(x):
            e, g = _energy_and_grad(x, r_curr)
            return e
            
        def grad_curr(x):
            e, g = _energy_and_grad(x, r_curr)
            return g
            
        res = minimize(
            obj_curr, x0, jac=grad_curr, method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x0 = res.x
        
    centers = x0.reshape(-1, 2)
    
    # Step 3: Final Polish
    # Try to maximize r further.
    # We can do a binary search for the max r that can be satisfied with low energy.
    # Or just run a few more iterations at the target r.
    
    # Let's check the current score
    score = _compute_score(centers)
    
    # If we are below target, try to push harder.
    # If we are above, great.
    
    # One more round of optimization with a slightly higher target to see if we can squeeze more.
    r_high = r_target * 1.01
    def obj_high(x):
        e, g = _energy_and_grad(x, r_high)
        return e
        
    def grad_high(x):
        e, g = _energy_and_grad(x, r_high)
        return g
        
    res_final = minimize(
        obj_high, x0, jac=grad_high, method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    centers_high = res_final.x.reshape(-1, 2)
    score_high = _compute_score(centers_high)
    
    if score_high > score:
        centers = centers_high
    else:
        # Re-optimize at the best r we found?
        # The score is determined by the min distance.
        # Let's just return the one with the higher score.
        pass
        
    # Safety check
    radii = np.linalg.norm(centers, axis=1)
    mask_out = radii > 1.0
    if np.any(mask_out):
        centers[mask_out] *= (1.0 / radii[mask_out])[:, np.newaxis]
        
    return centers
# EVOLVE-BLOCK-END
