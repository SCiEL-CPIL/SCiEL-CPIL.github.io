import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _relax(points, r_target, max_iters=2000):
    """
    Vectorized force-directed relaxation.
    Pushes points apart if distance < 2*r_target.
    Pushes points inward if distance from origin > 1 - r_target.
    """
    points = points.copy()
    n = len(points)
    # Initial learning rate, decays over time
    lr = 0.05 
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # 1. Pairwise forces
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        
        # Force magnitude: proportional to overlap
        # If d < 2*r, force = (2*r - d) * direction
        # We want to push i away from j if d_ij < 2r
        # F_i = sum_{j != i} (2r - d_ij) * (p_i - p_j) / d_ij  for d_ij < 2r
        # Note: (p_i - p_j) / d_ij is unit vector from j to i.
        
        # Vectorized calculation
        # Mask for pairs closer than 2*r
        mask = d < 2 * r_target
        # Avoid division by zero (diagonal is inf, so mask is False there)
        # d_inv = np.where(mask, 1.0 / d, 0.0)
        
        # Diff matrix: points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        # Force contribution per pair
        # shape (N, N, 2)
        # f_ij = (2r - d_ij) * diff_ij / d_ij
        # Only where mask is True
        
        # Efficient way:
        # overlap = np.where(mask, 2 * r_target - d, 0.0)
        # unit_vec = diff / d[:, :, np.newaxis] # Careful with 0s
        # Actually, let's just compute the force vector directly.
        
        # F_i = sum_j overlap_ij * (p_i - p_j) / d_ij
        # This is equivalent to:
        # F_i = sum_j (overlap_ij / d_ij) * (p_i - p_j)
        
        # Let's use a simpler vectorized approach that avoids explicit loops
        # but might be memory heavy for N=50? 50x50x2 is tiny.
        
        # Compute unit vectors
        # d is (N, N)
        # diff is (N, N, 2)
        
        # We only care about mask
        # force_vec = np.zeros_like(points)
        
        # Calculate pairwise forces
        # d_safe = np.where(d > 1e-12, d, 1.0)
        # unit_diff = diff / d_safe[:, :, np.newaxis]
        # overlap = np.where(mask, 2 * r_target - d, 0.0)
        # force_contrib = overlap[:, :, np.newaxis] * unit_diff
        # forces = np.sum(force_contrib, axis=1)
        
        # This is O(N^3) in memory if not careful, but N=50 is small.
        # Let's do it directly.
        
        d_safe = np.where(d > 1e-12, d, 1.0)
        unit_diff = diff / d_safe[:, :, np.newaxis]
        overlap = np.where(mask, 2 * r_target - d, 0.0)
        force_contrib = overlap[:, :, np.newaxis] * unit_diff
        forces = np.sum(force_contrib, axis=1)
        
        # 2. Boundary forces
        norms = np.hypot(points[:, 0], points[:, 1])
        # Push inward if norm > 1 - r_target
        # Force direction: -p_i / norm_i
        # Magnitude: (norm_i - (1 - r_target))
        
        # b_mask = norms > (1.0 - r_target)
        # b_overlap = np.where(b_mask, norms - (1.0 - r_target), 0.0)
        # b_forces = -b_overlap[:, np.newaxis] * points / norms[:, np.newaxis]
        # forces += b_forces
        
        # Vectorized boundary force
        b_mask = norms > (1.0 - r_target)
        b_overlap = np.where(b_mask, norms - (1.0 - r_target), 0.0)
        # Handle norm=0
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        b_forces = -b_overlap[:, np.newaxis] * points / norms_safe[:, np.newaxis]
        forces += b_forces
        
        # 3. Update positions
        # Add small noise to escape local minima?
        # noise = np.random.normal(0, 0.01 * lr, size=points.shape)
        
        points += lr * forces
        
        # Decay learning rate
        lr *= 0.995
        
        # Check convergence
        # if np.max(np.abs(forces)) < 1e-6: break
        
    return points

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    # Use a grid large enough
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _random_initial(seed=0):
    """Generate random initial configuration."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        # Uniform in disk
        r = np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Known best: 0.1258254895304
    # We will binary search for the max r
    
    # Initial configurations
    inits = []
    # Hexagonal lattice is a good start
    for spacing in [0.2, 0.22, 0.25]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            inits.append(pts)
    
    # Random starts
    for s in range(5):
        inits.append(_random_initial(s))
    
    # Binary search range
    # Lower bound: 0.1 (should be easy)
    # Upper bound: 0.14 (should be hard/impossible)
    lo = 0.1
    hi = 0.14
    
    # For each initial config, try to find the max r
    # But we have limited time.
    # Strategy:
    # 1. Try to achieve a known good score (0.1258) with the best init.
    # 2. If successful, try to push higher.
    # 3. If not, try other inits.
    
    # Actually, a simpler strategy:
    # Iterate through inits. For each, run a "push up" loop.
    # Start with r = 0.12.
    # Relax.
    # Check score.
    # If score > r, increase r.
    # If score < r, decrease r (or keep).
    
    # Let's do a coarse binary search on r for the best init first.
    # Or just iterate.
    
    # Time budget per init: ~20s
    # Total inits: 8
    # 8 * 20 = 160s. Too much.
    # Let's reduce inits or time per init.
    
    # Let's use 3 inits: Best Hex, Random 0, Random 1.
    # And a more aggressive relaxation.
    
    selected_inits = [inits[0], inits[3], inits[4]]
    
    for init in selected_inits:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Start with a target radius
        # We want to maximize the score.
        # Let's try to find the max feasible r for this init.
        
        # Binary search for max r
        lo_r = 0.10
        hi_r = 0.14
        
        best_r_for_init = lo_r
        best_pts_for_init = init.copy()
        
        for _ in range(10): # 10 iterations of binary search
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
                
            mid_r = (lo_r + hi_r) / 2.0
            
            # Relax with target mid_r
            relaxed = _relax(init, mid_r, max_iters=1000)
            
            # Check actual score
            score = _score(relaxed)
            
            if score > best_score:
                best_score = score
                best_points = relaxed.copy()
                
            # If we achieved mid_r (or close), it's feasible
            # Note: score is the *actual* min radius.
            # If score >= mid_r * 0.99, then mid_r is roughly feasible.
            # But we want to push as high as possible.
            
            if score >= mid_r * 0.99:
                lo_r = mid_r
                best_r_for_init = mid_r
                best_pts_for_init = relaxed.copy()
            else:
                hi_r = mid_r
        
        # After binary search, try to polish the best found so far
        # Use the best points found in this init's search
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Final polish with the best r found for this init
        # Relax with a slightly higher target to see if we can push more
        # Or just relax with the best r to stabilize
        
        # Let's do one more relaxation with the best r found
        final_relaxed = _relax(best_pts_for_init, best_r_for_init, max_iters=2000)
        final_score = _score(final_relaxed)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_relaxed.copy()
            
    # Fallback if no points found (should not happen)
    if best_points is None:
        best_points = _hexagonal_initial(0.22)
        if best_points is None:
            best_points = np.zeros((N, 2))
            
    # Ensure validity
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999
        
    return best_points
