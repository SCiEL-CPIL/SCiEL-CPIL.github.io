"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _radius(pts):
    """Compute the largest common radius for the given centres."""
    # Wall distance: min over i of (0.5 - max(|x_i|, |y_i|))
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_wall, min_pair)

def _pbd_step(pts, r, n_iters=100, dt=1.0):
    """
    Position-based dynamics step for circle packing.
    Enforces non-overlap (d_ij >= 2r) and wall constraints (dist_to_wall >= r).
    Returns updated pts.
    """
    pts = pts.copy()
    min_dist = 2 * r
    
    for _ in range(n_iters):
        # Pairwise collision resolution
        # We do iterative projection: for each pair, if too close, push apart.
        # To be more efficient, we can vectorize the check, but sequential projection is standard PBD.
        # However, N=50 is small, so even O(N^2) per iteration is fast.
        # Let's use a vectorized approach for finding violations, then resolve.
        # Actually, standard PBD often does sequential or parallel projection.
        # Parallel projection: compute all corrections, then apply.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)
        
        # Violations: d < min_dist
        # Correction vector for pair (i,j):
        # If d_ij < min_dist, we need to increase d_ij to min_dist.
        # Push i by (min_dist - d_ij)/2 in direction (p_i - p_j)/d_ij
        # Push j by (min_dist - d_ij)/2 in direction (p_j - p_i)/d_ij
        
        # Only consider pairs where d < min_dist
        mask = d < min_dist
        # Avoid division by zero
        d_safe = np.where(d > 1e-8, d, 1.0)
        
        # Correction magnitude
        # corr = (min_dist - d) / 2
        corr = np.where(mask, (min_dist - d) / 2.0, 0.0)
        
        # Unit vector
        # u_ij = (p_i - p_j) / d
        u = diff / d_safe[:, :, np.newaxis]
        
        # Total correction for point i: sum_j corr_ij * u_ij
        # Note: u_ij = (p_i - p_j)/d. So correction to p_i is corr * u.
        # Correction to p_j is -corr * u.
        
        # Sum over j for each i
        # C_i = sum_j corr_ij * u_ij
        C = np.sum(corr[:, :, np.newaxis] * u, axis=1)
        
        pts += C
        
        # Wall collision resolution
        # dist_to_wall = 0.5 - max(|x|, |y|)
        # We want dist_to_wall >= r => max(|x|, |y|) <= 0.5 - r
        # Let limit = 0.5 - r
        limit = 0.5 - r
        
        # For each point, if |x| > limit, push x towards 0.
        # If |y| > limit, push y towards 0.
        # If both, we need to be careful. The constraint is max(|x|,|y|) <= limit.
        # This is equivalent to x in [-limit, limit] AND y in [-limit, limit].
        # So we can simply clip x and y to [-limit, limit].
        
        pts = np.clip(pts, -limit, limit)
        
    return pts

def _local_search(pts, r, time_limit):
    """
    Perform local search to maximize radius starting from pts.
    Uses PBD to find feasible configuration for a target radius.
    Binary search on radius.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _radius(best_pts)
    
    # Initial radius estimate
    r_low = best_r
    # Upper bound: theoretical max for 50 circles is around 0.0714
    # Let's start a bit higher to see if we can improve
    r_high = 0.08
    
    # Binary search
    for _ in range(30):  # Max 30 steps
        if time.time() - t0 > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # Start from best_pts
        cand_pts = _pbd_step(best_pts, r_mid, n_iters=200)
        
        # Verify
        r_cand = _radius(cand_pts)
        
        if r_cand >= r_mid - 1e-6:
            # Feasible
            r_low = r_mid
            best_pts = cand_pts
            best_r = r_cand
        else:
            # Not feasible
            r_high = r_mid
        
        # Convergence check
        if r_high - r_low < 1e-7:
            break
    
    return best_pts, best_r

def construct():
    t0 = time.time()
    timeout = 145.0  # Safety margin
    
    # Initial guess: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    best_pts = grid.copy()
    best_r = _radius(best_pts)
    
    # Run local search from grid
    best_pts, best_r = _local_search(best_pts, best_r, timeout - 10)
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    
    while time.time() - t0 < timeout - 5:
        # Random initialization
        init_pts = rng.uniform(-0.5, 0.5, size=(N, 2))
        
        # Push apart to avoid coincident points
        for _ in range(20):
            d = np.sqrt(np.sum((init_pts[:, None, :] - init_pts[None, :, :]) ** 2, axis=2))
            np.fill_diagonal(d, np.inf)
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] < 0.05:
                        diff = init_pts[i] - init_pts[j]
                        norm = np.linalg.norm(diff)
                        if norm < 1e-8:
                            diff = rng.normal(size=2)
                            norm = np.linalg.norm(diff)
                        diff /= norm
                        init_pts[i] += 0.5 * diff * 0.05
                        init_pts[j] -= 0.5 * diff * 0.05
        init_pts = np.clip(init_pts, -0.5, 0.5)
        
        # Local search
        cand_pts, cand_r = _local_search(init_pts, _radius(init_pts), timeout - (time.time() - t0) - 5)
        
        if cand_r > best_r:
            best_pts = cand_pts
            best_r = cand_r
    
    return best_pts
# EVOLVE-BLOCK-END
