"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing: horizontal = s, vertical = s * sqrt(3)/2
        # With alternating offset of s/2
        # We need to fit in [-0.5, 0.5]^2
        
        # Let's compute the required spacing
        # Width: (n_cols - 1) * s + 2*r <= 1  =>  s + 2*r <= 1/(n_cols-1) if n_cols > 1
        # Actually, for hex: the width is (n_cols - 1) * s + 2*r
        # The height is (n_rows - 1) * s * sqrt(3)/2 + 2*r
        
        # We want to maximize r subject to:
        # (n_cols - 1) * s + 2*r <= 1
        # (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
        # s >= 2*r (non-overlapping in hex: nearest neighbor distance = s)
        
        # From hex geometry, nearest neighbor distance = s
        # So we need s >= 2*r
        
        # Let's solve: maximize r
        # s = 2*r (tight packing)
        # (n_cols - 1) * 2*r + 2*r <= 1  =>  2*r * n_cols <= 1  =>  r <= 1/(2*n_cols)
        # (n_rows - 1) * 2*r * sqrt(3)/2 + 2*r <= 1  =>  r * ((n_rows-1)*sqrt(3) + 2) <= 1  =>  r <= 1/((n_rows-1)*sqrt(3) + 2)
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        # Build the grid
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Check if all are in bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # Adjust: try to center the grid
            # Recompute with centering
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            # Now check if it fits
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _local_search(centers, max_iter=5000, step_scale=0.01):
    """Local search using gradient-free perturbation."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    tree = cKDTree(centers)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Find the most constrained pairs
        dists, idx = tree.query(centers, k=3)
        # For each point, the nearest neighbor distance
        nn_dists = dists[:, 1]
        nn_idxs = idx[:, 1]
        
        # The minimum distance pair
        min_idx = np.argmin(nn_dists)
        min_dist = nn_dists[min_idx]
        neighbor = nn_idxs[min_idx]
        
        # Also check wall constraints
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Determine which constraint is tighter
        # Effective radius is min(min_pair/2, min_wall)
        # If min_pair/2 < min_wall, the pair constraint is active
        # Otherwise the wall constraint is active
        
        # Perturb the most constrained points
        # Pick 2-3 points to move
        n_perturb = 2 + (it % 2)
        
        # Select points: the ones with smallest nn_dist and smallest wall_dist
        candidates = np.concatenate([
            np.argsort(nn_dists)[:n_perturb],
            np.argsort(wall_dists)[:n_perturb]
        ])
        candidates = np.unique(candidates[:n_perturb])
        
        # Generate random perturbations
        improvements = 0
        for _ in range(10):  # Try multiple perturbations per iteration
            trial = centers.copy()
            for idx in candidates:
                # Random direction
                angle = np.random.uniform(0, 2 * np.pi)
                # Step size proportional to current constraint tightness
                step = step_scale * (0.5 + np.random.rand())
                trial[idx, 0] += step * np.cos(angle)
                trial[idx, 1] += step * np.sin(angle)
            
            # Keep in bounds
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            
            r_trial = _compute_radius(trial)
            if r_trial > best_r + 1e-12:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                tree = cKDTree(centers)
                step_scale *= 0.95  # Reduce step size on improvement
                improvements += 1
        
        if improvements == 0:
            step_scale *= 0.99  # Slightly reduce step size
            if step_scale < 1e-6:
                break
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start with hexagonal grid
    centers = _generate_hex_grid()
    r = _compute_radius(centers)
    
    # Local search
    if _time_left() > 10.0:
        centers, r = _local_search(centers, max_iter=8000)
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
