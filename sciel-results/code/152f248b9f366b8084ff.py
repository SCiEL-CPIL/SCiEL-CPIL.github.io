"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform


def _compute_radius(pts):
    """Compute the maximum valid radius for given points in [-1/2, 1/2]^3."""
    if len(pts) < 2:
        return 0.5
    # Min distance between any pair of centers / 2
    d = pdist(pts)
    min_pair_dist = d.min() / 2.0
    # Min distance from any center to any wall
    # For cube [-1/2, 1/2], distance to wall = 1/2 - max(|x|, |y|, |z|)
    max_coord = np.abs(pts).max(axis=1)
    wall_dist = (0.5 - max_coord).min()
    return min(min_pair_dist, wall_dist)


def _generate_bcc_points(n):
    """Generate points from BCC lattice, clipped to cube, take first n."""
    # BCC lattice: simple cubic + body center
    # In unit cube [-1/2, 1/2]^3, BCC with spacing a has points at
    # (i*a, j*a, k*a) and ((i+0.5)*a, (j+0.5)*a, (k+0.5)*a)
    # We want to fit n points. Let's try different spacings.
    
    # For n=100, a 5x5x4 grid gives 100 points in simple cubic
    # But BCC is denser. Let's use a 4x4x4 BCC = 4*4*4*2 = 128 points, take 100
    
    # Try to find good lattice parameters
    # BCC in cube: points at integer coords and half-integer coords scaled
    
    # Let's use a simpler approach: generate BCC points in a larger region,
    # then select the n that fit best in the cube
    
    # For a BCC lattice with conventional cell side a, points per cell = 2
    # In cube of side 1, we can fit roughly (1/a)^3 cells, so ~2*(1/a)^3 points
    # For 100 points: 2*(1/a)^3 ≈ 100 → (1/a)^3 ≈ 50 → 1/a ≈ 3.68 → a ≈ 0.272
    
    # Let's generate BCC points with various spacings and pick the best set
    
    best_pts = None
    best_r = 0
    
    # Try a range of lattice spacings
    for a in np.linspace(0.25, 0.35, 20):
        # Generate BCC points in region [-1/2, 1/2]
        n_cells = int(1.0 / a)
        # Points from simple cubic sublattice
        sc_pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + i * a
                    y = -0.5 + j * a
                    z = -0.5 + k * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        sc_pts.append([x, y, z])
        
        # Points from body center sublattice
        bc_pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + (i + 0.5) * a
                    y = -0.5 + (j + 0.5) * a
                    z = -0.5 + (k + 0.5) * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        bc_pts.append([x, y, z])
        
        all_pts = sc_pts + bc_pts
        if len(all_pts) < n:
            continue
        
        # Select n points that maximize minimum distance
        # Simple approach: take first n, or try to select optimally
        # For now, take a random subset and keep the best
        np.random.seed(42)
        all_pts_arr = np.array(all_pts)
        
        # Try multiple random selections
        best_subset_r = 0
        for _ in range(50):
            if len(all_pts_arr) <= n:
                subset = all_pts_arr
            else:
                idx = np.random.choice(len(all_pts_arr), n, replace=False)
                subset = all_pts_arr[idx]
            
            r = _compute_radius(subset)
            if r > best_subset_r:
                best_subset_r = r
                best_subset = subset
        
        if best_subset_r > best_r:
            best_r = best_subset_r
            best_pts = best_subset
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: simple grid
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _local_optimize(pts, max_iter=1000, lr=0.01, deadline=None):
    """Local optimization using gradient-free approach to maximize minimum radius."""
    pts = pts.copy()
    n = len(pts)
    
    if deadline is not None and time.time() > deadline:
        return pts
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_d = dist[i, j]
        
        # Also check wall distances
        wall_dists = 0.5 - np.abs(pts).max(axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_d = wall_dists[min_wall_idx]
        
        # Determine which constraint is active
        if min_d / 2.0 < min_wall_d:
            # Pair constraint active: move i and j apart
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            
            # Move i in direction, j in -direction
            pts[i] += lr * direction
            pts[j] -= lr * direction
        else:
            # Wall constraint active: move point toward center
            p = pts[min_wall_idx]
            # Find which wall is closest
            for k in range(3):
                if np.abs(p[k]) == np.abs(p).max():
                    # Move away from this wall
                    if p[k] > 0:
                        pts[min_wall_idx, k] -= lr
                    else:
                        pts[min_wall_idx, k] += lr
                    break
        
        # Project back into cube
        pts = np.clip(pts, -0.5, 0.5)
        
        # Check improvement
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        else:
            # Revert
            pts = best_pts.copy()
    
    return best_pts


def _simulated_annealing(pts, max_iter=5000, deadline=None):
    """Simulated annealing to escape local minima."""
    pts = pts.copy()
    n = len(pts)
    T = 0.05
    cooling = 0.995
    
    if deadline is not None and time.time() > deadline:
        return pts
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random perturbation
        idx = np.random.randint(n)
        delta = np.random.randn(3) * T
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        diff = new_r - best_r
        
        if diff > 0 or np.random.random() < np.exp(diff / (T + 1e-10)):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        
        T *= cooling
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate initial BCC-based points
    pts = _generate_bcc_points(N)
    
    # Local optimization
    pts = _local_optimize(pts, max_iter=500, lr=0.005, deadline=deadline)
    
    # Simulated annealing for global improvement
    if time.time() < deadline - 10:
        pts = _simulated_annealing(pts, max_iter=3000, deadline=deadline)
    
    # Final local polish
    if time.time() < deadline - 5:
        pts = _local_optimize(pts, max_iter=200, lr=0.001, deadline=deadline)
    
    # Ensure validity
    pts = np.clip(pts, -0.5, 0.5)
    
    return pts
# EVOLVE-BLOCK-END
