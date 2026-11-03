"""Rewritten program: BCC lattice starts + multi-restart local optimization with deadline guard."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_FRAC = 0.90  # finish by 90% of wall-clock to be safe


def _compute_radius(centres):
    """Return the largest common radius for given centres."""
    # Wall distance: for each sphere i, distance to wall = min over k of (CUBE_HALF - |c_ik|)
    dist_to_wall = CUBE_HALF - np.abs(centres)  # shape (N,3)
    r_wall_per_sphere = np.min(dist_to_wall, axis=1)  # shape (N,)
    r_wall = np.min(r_wall_per_sphere)

    # Pairwise distances
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0

    return min(r_wall, r_pair)


def _bcc_lattice(n_spheres):
    """Generate a BCC-like lattice configuration for n_spheres spheres in unit cube."""
    # BCC unit cell has 2 atoms: corners + body center
    # In a supercell of m x m x m BCC cells, we have 2*m^3 positions
    # For m=3: 54 positions, pick 50
    m = 3
    positions = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner-like position (centered in subcell)
                x0 = (ix + 0.5) / m - 0.5
                y0 = (iy + 0.5) / m - 0.5
                z0 = (iz + 0.5) / m - 0.5
                positions.append([x0, y0, z0])
                # Body center offset
                dx = 1.0 / (2 * m)
                positions.append([x0 + dx, y0 + dx, z0 + dx])
    positions = np.array(positions[:n_spheres])
    return positions


def _random_perturbation(base, sigma=0.02, rng=None):
    """Add Gaussian perturbation and clip to cube."""
    if rng is None:
        rng = np.random.default_rng()
    perturbed = base + rng.normal(0, sigma, size=base.shape)
    perturbed = np.clip(perturbed, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
    return perturbed


def _local_optimize(centres, max_iter=5000, lr_init=0.01, deadline=None):
    """
    Local optimization to maximize the minimum radius.
    Uses vectorized force calculation for speed.
    """
    centres = centres.copy()
    n = len(centres)
    best_r = _compute_radius(centres)
    best_centres = centres.copy()

    lr = lr_init
    # Pre-allocate indices for pairwise operations if needed, but vectorized ops are better
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Compute pairwise distances
        d = cdist(centres, centres)
        np.fill_diagonal(d, np.inf)

        # Find the closest pair for tracking best_r
        min_d = np.min(d)
        
        # Wall distances
        dist_to_wall = CUBE_HALF - np.abs(centres)
        r_wall_per_sphere = np.min(dist_to_wall, axis=1)
        r_wall = np.min(r_wall_per_sphere)

        r_pair = min_d / 2.0
        r = min(r_wall, r_pair)

        if r > best_r + 1e-12:
            best_r = r
            best_centres = centres.copy()

        # Compute forces vectorized
        forces = np.zeros_like(centres)

        # Pairwise repulsion
        # We want to push apart points that are close.
        # Force ~ 1/d^2 * direction
        # To avoid division by zero, we use a softening epsilon
        eps = 1e-6
        # d is (N, N)
        # We only care about i < j, but vectorized ops on full matrix are easier if we handle symmetry
        # Let's compute diff matrix
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :] # (N, N, 3)
        
        # Inverse squared distance
        # d^2 + eps^2
        d2 = d * d + eps * eps
        
        # Force magnitude: 1/d^2
        # We apply a cutoff to limit long-range forces which are not needed for local packing
        # Threshold ~ 0.5 is reasonable for unit cube with 50 spheres
        mask = d < 0.5
        # Set forces to 0 where mask is False
        # F_ij = diff_ij / d_ij^3 * (some scaling)
        # Actually, force on i due to j is proportional to (c_i - c_j) / |c_i - c_j|^3
        # Let's use a softer potential to prevent explosion: F = diff / (d^2 + eps)^1.5 ? 
        # Or just clip d to eps.
        
        # Safe distance for division
        d_safe = np.maximum(d, eps)
        
        # Unit vectors
        # u_ij = diff / d
        # Force on i from j: u_ij / d^2 = diff / d^3
        # We sum over j != i
        
        # To save memory and time, we can compute the sum for i<j and add to i, subtract from j
        # But vectorized full matrix is (N, N, 3). N=50 -> 12500 vectors. This is small.
        
        # Compute raw force vectors
        # F_raw[i, j] = diff[i, j] / d_safe[i, j]^3
        # Apply mask: if d > threshold, force is 0.
        
        # d_safe^3
        d3 = d_safe ** 3
        
        # Force vectors
        F_vec = diff / d3[:, :, np.newaxis]
        
        # Apply mask: zero out forces where d >= 0.5
        # Also zero out diagonal (self-interaction)
        mask_3d = mask[:, :, np.newaxis]
        F_vec = np.where(mask_3d, F_vec, 0.0)
        
        # Sum over j (axis 1) to get net force on i
        # Sum F_vec[i, j] for all j. Since F_vec[i,j] = -F_vec[j,i], this works.
        # Note: F_vec[i,i] is 0 because diff[i,i]=0.
        forces_pair = np.sum(F_vec, axis=1)
        
        # Wall repulsion
        # Push away from walls if close
        wall_thresh = 0.2
        # dist_to_wall is (N, 3)
        # If dist < wall_thresh, apply force towards center
        # Force direction: -sign(c_ik)
        # Magnitude: (wall_thresh - dist) / dist ? Or just linear?
        
        # Let's use a simple linear repulsion from walls
        # f_wall = max(0, wall_thresh - dist)
        # Direction: -sign(c)
        
        wall_forces = np.zeros_like(centres)
        for k in range(3):
            # dist_to_wall[:, k] is distance to nearest wall in axis k
            # But we need to know which wall.
            # If c > 0, wall is at 0.5, force is -x.
            # If c < 0, wall is at -0.5, force is +x.
            # dist = 0.5 - |c|
            # Force magnitude = max(0, wall_thresh - dist)
            # Force direction = -sign(c)
            
            dist_k = dist_to_wall[:, k]
            mag = np.maximum(0, wall_thresh - dist_k)
            # If c is 0, sign is 0, force is 0? No, we need a direction.
            # If c=0, it's in the middle, no wall force needed if dist is large.
            # If dist < thresh, c is close to wall, so |c| is close to 0.5.
            # So sign(c) is well defined (non-zero).
            signs = np.sign(centres[:, k])
            # If sign is 0, force is 0.
            wall_forces[:, k] = -signs * mag

        forces = forces_pair + wall_forces

        # Update positions
        centres += lr * forces
        centres = np.clip(centres, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)

        # Anneal learning rate
        lr = max(lr_init * (0.999 ** it), 1e-6)

    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRAC

    rng = np.random.default_rng(42)

    # Generate multiple initial configurations
    configs = []

    # BCC lattice
    bcc = _bcc_lattice(N)
    configs.append(bcc)

    # Perturbed BCC variants
    for sigma in [0.01, 0.02, 0.03, 0.05]:
        perturbed = _random_perturbation(bcc, sigma=sigma, rng=rng)
        configs.append(perturbed)

    # Simple cubic grid (4x4x4 = 64, take 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    configs.append(grid[:N])

    # Random uniform
    random_cfg = rng.uniform(-CUBE_HALF + 0.02, CUBE_HALF - 0.02, size=(N, 3))
    configs.append(random_cfg)

    best_centres = configs[0]
    best_r = _compute_radius(best_centres)

    # Allocate time budget per config
    n_configs = len(configs)
    # Reserve 20% for final polish
    time_per_config = (deadline - start_time - (deadline - start_time) * 0.2) / n_configs
    # But we must respect the global deadline.
    
    for idx, cfg in enumerate(configs):
        # Check if we have enough time for a reasonable optimization
        current_time = time.time()
        # If less than 2 seconds left, break
        if current_time > deadline - 2:
            break
        
        # Local optimization
        # Use a dynamic deadline for this config: min(global_deadline - 2, current_time + time_per_config)
        local_deadline = min(deadline - 2, current_time + time_per_config)
        
        opt_centres, opt_r = _local_optimize(
            cfg,
            max_iter=10000, # High limit, time will cut it
            lr_init=0.01,
            deadline=local_deadline
        )

        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres

    # Final polish: one more pass with tighter settings
    if time.time() < deadline - 5:
        polished, polished_r = _local_optimize(
            best_centres,
            max_iter=5000,
            lr_init=0.002,
            deadline=deadline - 1
        )
        if polished_r > best_r:
            best_r = polished_r
            best_centres = polished

    return best_centres
