"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START


def _compute_radius(centers):
    """Compute the valid common radius for a set of centers in [-1/2,1/2]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return 0.5 * min(min_pair, 2.0 * min_wall)


def _soft_min_radius(centers, beta=50.0):
    """Smooth surrogate for the min radius using log-sum-exp."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e10)
    # Soft min of pairwise distances / 2
    pair_vals = dists / 2.0
    pair_vals = pair_vals[~np.eye(n, dtype=bool)]
    # Soft min via log-sum-exp (negative log-sum-exp gives soft min)
    m = np.min(pair_vals)
    soft_min_pair = m - np.log(np.sum(np.exp(m - pair_vals)) / len(pair_vals))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_flat = wall_dists.flatten()
    mw = np.min(wall_flat)
    soft_min_wall = mw - np.log(np.sum(np.exp(mw - wall_flat)) / len(wall_flat))
    
    # Combine: soft min of (soft_min_pair, soft_min_wall)
    vals = np.array([soft_min_pair, soft_min_wall])
    mm = np.min(vals)
    soft_min_combined = mm - np.log(np.sum(np.exp(mm - vals)) / len(vals))
    return soft_min_combined


def _make_fcc_packing():
    """Create an FCC lattice packing of 50 spheres in the unit cube."""
    # FCC lattice points in a cube: use face-centered cubic arrangement
    # For 50 spheres, we'll use a grid of FCC unit cells
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    
    # Try different scales to fit 50 spheres
    best_centers = None
    best_r = 0.0
    
    # FCC unit cell has 4 atoms; we need 50/4 = 12.5 cells
    # Use a 3x3x2 grid of FCC cells = 36 atoms, then add more
    # Or use 2x3x3 = 36, or 3x3x3 = 54 (take 50)
    
    for nx, ny, nz in [(3, 3, 3), (2, 3, 3), (3, 2, 3), (3, 3, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)]:
        total = nx * ny * nz * 4
        if total < 50:
            continue
        
        # FCC lattice vectors
        a = 1.0 / max(nx, ny, nz)  # scale to fit in cube
        # Actually, let's think more carefully
        # We want to pack in [-0.5, 0.5]^3
        # FCC primitive vectors in a cubic cell of side a:
        #   v1 = (a/2, a/2, 0), v2 = (a/2, 0, a/2), v3 = (0, a/2, a/2)
        # Or use the conventional cubic cell with 4 points
        
        # Let's use a simpler approach: generate FCC points in a box
        # and scale to fit
        
        # Conventional FCC cell: points at corners and face centers
        # In a cell of side a, points at:
        # (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
        # Plus translations
        
        # For a grid of nx x ny x nz conventional cells:
        # Total points = 4 * nx * ny * nz (with some at boundaries)
        
        pts = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Corner of this cell
                    x0 = i * a
                    y0 = j * a
                    z0 = k * a
                    # 4 FCC points in this cell
                    pts.append([x0, y0, z0])
                    pts.append([x0 + a/2, y0 + a/2, z0])
                    pts.append([x0 + a/2, y0, z0 + a/2])
                    pts.append([x0, y0 + a/2, z0 + a/2])
        
        pts = np.array(pts)
        # Remove duplicates (boundary points shared between cells)
        pts_unique, _ = np.unique(np.round(pts, 10), axis=0, return_index=True)
        pts_unique = pts_unique[:50]
        
        if len(pts_unique) < 50:
            continue
        
        # Center and scale to fit in [-0.5, 0.5]^3
        center = np.mean(pts_unique, axis=0)
        pts_centered = pts_unique - center
        max_extent = np.max(np.abs(pts_centered))
        if max_extent > 0:
            pts_centered = pts_centered / max_extent * 0.45  # leave margin
        
        r = _compute_radius(pts_centered)
        if r > best_r:
            best_r = r
            best_centers = pts_centered.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers


def _force_directed_optimize(centers, max_time=100.0, beta=50.0):
    """Optimize sphere packing using force-directed method with soft-min objective."""
    start_time = time.time()
    centers = np.array(centers, dtype=np.float64, copy=True)
    n = centers.shape[0]
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Adaptive step size
    step = 0.01
    patience = 0
    max_patience = 500
    
    while time.time() - start_time < max_time:
        if time.time() > DEADLINE - 5:
            break
        
        # Compute forces: push apart close pairs, pull toward center
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Force: inverse distance squared (repulsion)
        # Only act on pairs closer than some threshold
        threshold = 0.3
        mask = (dists < threshold) & (dists > 1e-10)
        forces = np.zeros_like(centers)
        
        # Vectorized force computation
        inv_d2 = np.zeros_like(dists)
        inv_d2[mask] = 1.0 / (dists[mask] ** 2)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Wall constraint: push away from walls
        for axis in range(3):
            # Left wall (x < -0.5)
            left_margin = centers[:, axis] + 0.5
            right_margin = 0.5 - centers[:, axis]
            # Strong penalty near walls
            wall_force = np.zeros(n)
            wall_force[left_margin < 0.05] -= 1.0 / (left_margin[left_margin < 0.05] + 1e-6) ** 2
            wall_force[right_margin < 0.05] += 1.0 / (right_margin[right_margin < 0.05] + 1e-6) ** 2
            forces[:, axis] += wall_force
        
        # Normalize forces
        force_norms = np.sqrt(np.sum(forces ** 2, axis=1))
        max_force = np.max(force_norms)
        if max_force > 0:
            forces = forces / max_force
        
        # Update positions
        new_centers = centers + step * forces
        
        # Clip to cube with small margin
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        # Compute new radius
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r + 1e-10:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers
            patience = 0
            # Slowly decrease step
            step *= 0.995
        else:
            patience += 1
            if patience > max_patience:
                break
            # Reduce step on failure
            step *= 0.95
            # Try a small random perturbation
            if patience % 50 == 0:
                centers = best_centers + np.random.randn(n, 3) * 0.005
                centers = np.clip(centers, -0.499, 0.499)
        
        # Early exit if step is too small
        if step < 1e-8:
            break
    
    return best_centers, best_r


def _polish(centers, max_time=20.0):
    """High-precision polish pass with tighter tolerance."""
    start_time = time.time()
    centers = np.array(centers, dtype=np.float64, copy=True)
    n = centers.shape[0]
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = 0.005
    while time.time() - start_time < max_time:
        if time.time() > DEADLINE - 3:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Only optimize the closest few pairs
        # Find pairs with distance < 2*r + small_margin
        r_est = best_r
        threshold = 2.0 * r_est + 0.02
        mask = (dists < threshold) & (dists > 1e-10)
        
        if not np.any(mask):
            break
        
        forces = np.zeros_like(centers)
        inv_d2 = np.zeros_like(dists)
        inv_d2[mask] = 1.0 / (dists[mask] ** 2)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Wall forces
        for axis in range(3):
            left_margin = centers[:, axis] + 0.5
            right_margin = 0.5 - centers[:, axis]
            wall_force = np.zeros(n)
            wall_force[left_margin < 0.02] -= 1.0 / (left_margin[left_margin < 0.02] + 1e-6) ** 2
            wall_force[right_margin < 0.02] += 1.0 / (right_margin[right_margin < 0.02] + 1e-6) ** 2
            forces[:, axis] += wall_force
        
        force_norms = np.sqrt(np.sum(forces ** 2, axis=1))
        max_force = np.max(force_norms)
        if max_force > 0:
            forces = forces / max_force
        
        new_centers = centers + step * forces
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers
            step *= 0.99
        else:
            step *= 0.9
            if step < 1e-10:
                break
    
    return best_centers, best_r


def construct():
    """Construct 50 sphere centers in the unit cube to maximize common radius."""
    start_time = time.time()
    
    # Generate FCC lattice initial configuration
    centers = _make_fcc_packing()
    
    # Force-directed optimization
    centers, r = _force_directed_optimize(centers, max_time=110.0)
    
    # Check deadline
    if time.time() > DEADLINE - 25:
        return centers
    
    # Polish pass
    centers, r = _polish(centers, max_time=20.0)
    
    return centers
# EVOLVE-BLOCK-END
