"""Solve 100 equal spheres in unit cube: direct construction with optimization."""

import numpy as np
import time

N = 100
CUBE_HALF = 0.5
DEADLINE = 115  # seconds, safety margin before 120s limit


def compute_radius(centres: np.ndarray) -> float:
    """Compute the common radius r for given centres."""
    # Wall distance
    max_c = np.abs(centres).max(axis=1)
    wall_min = CUBE_HALF - max_c
    r_wall = wall_min.min()
    
    # Pairwise distances
    n = len(centres)
    if n < 2:
        return r_wall
    
    # Use scipy's cdist or manual
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    # Upper triangle
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(r_wall, min_pair)


def initial_grid(n: int = N) -> np.ndarray:
    """Create initial grid arrangement."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - CUBE_HALF
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n].astype(float)


def fea_optimize(centres: np.ndarray, target_r: float, 
                  iters: int = 200, lr: float = 0.001) -> tuple:
    """
    Force-exclusion algorithm to push spheres apart while staying in cube.
    Returns (new_centres, min_pair_distance/2, min_wall).
    """
    from scipy.spatial.distance import cdist
    
    c = centres.copy().astype(float)
    n = len(c)
    
    # Compute current state
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
    current_r = min(min_pair, wall_min)
    
    if current_r >= target_r:
        return c, min_pair, (CUBE_HALF - np.abs(c).max(axis=1)).min()
    
    # FEA: iteratively push apart overlapping spheres
    for it in range(iters):
        if time.time() > DEADLINE:
            break
            
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        
        # Check if we're already at target
        min_pair = dists.min() / 2.0
        wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
        if min(min_pair, wall_min) >= target_r:
            break
        
        # Build force vectors (vectorized)
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (only for pairs closer than 2*target_r)
        # Vectorized: compute all pairwise differences and distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n, n, 3)
        d = np.linalg.norm(diff, axis=2)  # (n, n)
        np.fill_diagonal(d, np.inf)
        
        # Mask for pairs that need pushing apart
        mask = d < 2.0 * target_r
        mask &= (d > 1e-10)
        
        # Force magnitude: (2*target_r - d) / 2, zero where not needed
        force_mag = np.where(mask, (2.0 * target_r - d) / 2.0, 0.0)
        
        # Unit vectors (safe division)
        safe_d = np.where(d > 1e-10, d, 1.0)
        unit_vec = diff / safe_d[:, :, np.newaxis]
        
        # Apply forces
        weighted = force_mag[:, :, np.newaxis] * unit_vec
        forces = weighted.sum(axis=1)
        
        # Wall repulsion (vectorized)
        # Positive wall
        pos_viol = np.maximum(c + target_r - CUBE_HALF, 0.0)
        forces -= pos_viol
        # Negative wall  
        neg_viol = np.maximum(-CUBE_HALF - (c - target_r), 0.0)
        forces += neg_viol
        
        c += lr * forces
        
        # Clip to cube (with small margin)
        c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    # Final metrics
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
    
    return c, min_pair, wall_min


def local_search(centres: np.ndarray, iters: int = 50, step_size: float = 0.02) -> np.ndarray:
    """
    Simple local search: try small random perturbations and keep if radius improves.
    Uses efficient radius computation by tracking nearest neighbors.
    """
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    
    c = centres.copy().astype(float)
    n = len(c)
    
    def radius_of(arr):
        dists = cdist(arr, arr)
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0
        wall_min = (CUBE_HALF - np.abs(arr).max(axis=1)).min()
        return min(min_pair, wall_min)
    
    best_r = radius_of(c)
    
    # Build KD-tree for efficient neighbor queries
    tree = cKDTree(c)
    
    for it in range(iters):
        if time.time() > DEADLINE:
            break
            
        # Pick a random sphere and try a small move
        i = np.random.randint(n)
        original = c[i].copy()
        
        # Try a few random directions
        improved = False
        for _ in range(10):
            delta = np.random.randn(3) * step_size
            c[i] = original + delta
            
            # Keep in cube
            c[i] = np.clip(c[i], -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
            
            r = radius_of(c)
            if r > best_r:
                best_r = r
                improved = True
                # Update tree with new position
                tree.data[i] = c[i]
            else:
                c[i] = original
                break
        
    return c


def smooth_optimize(centres: np.ndarray, iters: int = 100, lr: float = 0.001, 
                     soft_beta: float = 50.0) -> np.ndarray:
    """
    Smooth surrogate optimization using log-sum-exp soft-min.
    Maximizes the smooth approximation of the minimum radius.
    """
    from scipy.spatial.distance import cdist
    
    c = centres.copy().astype(float)
    n = len(c)
    
    def smooth_radius(arr, beta=soft_beta):
        """Compute smooth approximation of min radius using log-sum-exp."""
        # Wall distances
        wall_dists = CUBE_HALF - np.abs(arr).max(axis=1)
        
        # Pairwise distances (half)
        dists = cdist(arr, arr)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Combine all constraints
        # We want min(wall_dists, pair_dists)
        # Use soft-min via log-sum-exp: -log(sum(exp(-beta*x))) / beta
        all_dists = np.concatenate([wall_dists, pair_dists[np.triu_indices(n, k=1)]])
        
        # Soft-min approximation
        scaled = -beta * all_dists
        # Numerical stability
        max_scaled = scaled.max()
        log_sum = max_scaled + np.log(np.exp(scaled - max_scaled).sum())
        soft_min = -log_sum / beta
        
        return soft_min
    
    best_r = smooth_radius(c)
    
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        
        # Gradient via finite differences (expensive but simple)
        # Use central differences for better accuracy
        grad = np.zeros_like(c)
        eps = 1e-5
        
        # For efficiency, only compute gradient for a subset of spheres per iteration
        # Or use full gradient if time permits
        for i in range(n):
            for k in range(3):
                c_plus = c.copy()
                c_plus[i, k] += eps
                c_plus = np.clip(c_plus, -CUBE_HALF, CUBE_HALF)
                
                c_minus = c.copy()
                c_minus[i, k] -= eps
                c_minus = np.clip(c_minus, -CUBE_HALF, CUBE_HALF)
                
                f_plus = smooth_radius(c_plus)
                f_minus = smooth_radius(c_minus)
                grad[i, k] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        c += lr * grad
        c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        
        new_r = smooth_radius(c)
        if new_r < best_r:
            # Revert if it got worse
            # Keep the previous position (simplified)
            pass
        best_r = max(best_r, new_r)
    
    return c


def construct() -> np.ndarray:
    """
    Construct centres for 100 equal spheres in unit cube.
    Uses grid initialization + FEA optimization + local search.
    """
    t_start = time.time()
    deadline = DEADLINE
    
    # Start with grid
    c = initial_grid(N)
    r_init = compute_radius(c)
    
    # Target radius: slightly above current best known
    # Best known: ~0.11138, try to beat that
    target_r = 0.1115
    
    # Phase 1: FEA to expand spacing (vectorized version is faster)
    c, _, _ = fea_optimize(c, target_r, iters=500, lr=0.003)
    
    if time.time() > deadline:
        # Return whatever we have
        return np.clip(c, -CUBE_HALF, CUBE_HALF)
    
    r_after_fea = compute_radius(c)
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 30:
        c = local_search(c, iters=300, step_size=0.01)
        r_after_ls = compute_radius(c)
    
    # Final validation
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return c
