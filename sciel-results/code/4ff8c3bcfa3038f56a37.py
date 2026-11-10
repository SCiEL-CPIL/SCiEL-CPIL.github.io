"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    # Use scipy if available for speed, else fallback
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
    except ImportError:
        # Fallback: compute all pairwise distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centres[i] - centres[j])
                if d < min_dist:
                    min_dist = d
    return min(min_dist / 2.0, min_wall)


def _grid_packing():
    """Create a square grid packing with optimal dimensions for N points."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(10, 25):
        # For hexagonal packing, alternating rows have offset
        # Row 0: n_cols points, Row 1: n_cols - 1 points (or vice versa)
        # Let's use a pattern where we can fit N points
        
        # Try both orientations: start with full row or offset row
        for start_offset in [0, 1]:
            # Estimate columns needed
            # If start_offset=0: rows 0,2,4,... have n_cols, rows 1,3,5,... have n_cols-1
            # If start_offset=1: rows 0,2,4,... have n_cols-1, rows 1,3,5,... have n_cols
            
            # Let's try to find n_cols such that total points >= N
            # For simplicity, let's try a range of n_cols
            for n_cols in range(10, 25):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing s
                # Horizontal: n_cols * s <= 1 (for full rows)
                # Vertical: (n_rows - 1) * s * sqrt(3)/2 + s <= 1 (approx)
                # Actually, vertical extent is (n_rows - 1) * s * sqrt(3)/2
                # And we need to center it
                
                # Width needed: n_cols * s for full rows
                # But offset rows are shifted by s/2, so they fit in same width if n_cols-1 points
                # Let's say s = 1.0 / n_cols (horizontal constraint)
                # Vertical: (n_rows - 1) * s * sqrt(3)/2 <= 1
                # So s <= 2 / ((n_rows - 1) * sqrt(3)) if n_rows > 1
                
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                r = s / 2.0
                
                if r <= best_r:
                    continue
                
                # Create the grid
                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2  # Center vertically
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        # Full row: n_cols points
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        # Offset row: n_cols - 1 points
                        x_start = -0.5 + s  # Shift by s/2 from full row start
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    return _grid_packing()


def _local_optimization(centres, max_iters=200, step_size=0.02):
    """
    Interior-point optimization with smooth log-barrier for max-min packing.
    Uses a smooth surrogate for the nonsmooth min function via log-sum-exp,
    and enforces constraints through barrier terms.
    """
    best_centres = centres.copy()
    current_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_r = _compute_radius(current_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10)  # Leave 10s buffer for final checks
    
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        cdist = None

    # Smoothness parameter for log-sum-exp (controls how "sharp" the min is)
    # Smaller beta = smoother, larger beta = sharper (closer to true min)
    beta_start = 50.0
    beta_end = 500.0
    
    # Barrier weight
    barrier_weight = 1.0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Anneal beta from smooth to sharp
        progress = it / max_iters
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Compute smooth objective and gradient
        # We want to maximize: min over all pairs and walls of distances
        # Smooth surrogate: -log(sum(exp(-beta * d))) / beta approximates min(d)
        
        if cdist is not None:
            dists = cdist(current_centres, current_centres)
            np.fill_diagonal(dists, np.inf)
            
            # Wall distances for each point
            wall_dists = np.stack([
                0.5 - current_centres[:, 0],
                0.5 + current_centres[:, 0],
                0.5 - current_centres[:, 1],
                0.5 + current_centres[:, 1]
            ], axis=1)
            
            # All distances: pairwise + wall
            # For smooth min, we combine all constraints
            # Pairwise: d_ij / 2 (radius constraint)
            # Wall: d_wall (radius constraint)
            
            # Smooth min of all pairwise half-distances and wall distances
            # Use log-sum-exp trick: min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
            
            # Collect all relevant distances
            # To avoid memory issues, compute in chunks
            smooth_obj = 0.0
            grad = np.zeros_like(current_centres)
            
            # Pairwise contributions
            # For each pair (i,j), distance d_ij, contributes to smooth min
            # Gradient of -log(sum(exp(-beta*d))) w.r.t. position
            # = sum_j [ (beta * exp(-beta*d_ij) / sum_k exp(-beta*d_ik)) * (c_i - c_j)/d_ij ]
            
            # Simplified: use weighted average of directions to nearest neighbors
            # This is an approximation but captures the essential structure
            
            # Compute pairwise distances (upper triangle only for efficiency)
            i_idx, j_idx = np.triu_indices(N, k=1)
            pair_dists = dists[i_idx, j_idx]
            
            # For smooth min, weight each pair by exp(-beta * d)
            # Normalize to get probability-like weights
            weights = np.exp(-beta * pair_dists)
            weights /= weights.sum()
            
            # Gradient contribution from each pair to points i and j
            # d/dx_i [-log(sum exp(-beta*d))] = sum_j w_j * beta * (x_i - x_j)/d_ij
            # Simplified: direction from i to j weighted by w_j
            
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                d = pair_dists[k]
                if d < 1e-10:
                    continue
                w = weights[k] * beta
                direction = (current_centres[i] - current_centres[j]) / d
                grad[i] += w * direction
                grad[j] -= w * direction
            
            # Wall contributions
            # For each point, 4 wall distances
            # Smooth min over all wall distances
            all_wall_dists = wall_dists.flatten()
            wall_weights = np.exp(-beta * all_wall_dists)
            wall_weights /= wall_weights.sum()
            
            # Reshape to (N, 4)
            wall_weights_2d = wall_weights.reshape(N, 4)
            
            # Gradient from walls: push away from nearest wall
            # Left wall: x < 0, push right (+x)
            # Right wall: x > 0, push left (-x)  
            # Bottom wall: y < 0, push up (+y)
            # Top wall: y > 0, push down (-y)
            
            wall_grad = np.zeros_like(current_centres)
            for i in range(N):
                # Left wall weight
                wall_grad[i, 0] += wall_weights_2d[i, 0] * beta
                # Right wall weight
                wall_grad[i, 0] -= wall_weights_2d[i, 1] * beta
                # Bottom wall weight
                wall_grad[i, 1] += wall_weights_2d[i, 2] * beta
                # Top wall weight
                wall_grad[i, 1] -= wall_weights_2d[i, 3] * beta
            
            grad += wall_grad
            
            # Compute smooth objective value (for tracking)
            # min ≈ -1/beta * log(sum(exp(-beta*d)))
            # We'll just track the actual radius separately
            
        else:
            # Fallback: simple gradient estimate
            grad = np.zeros_like(current_centres)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(current_centres[i] - current_centres[j])
                    if d < 1e-10:
                        continue
                    direction = (current_centres[i] - current_centres[j]) / d
                    grad[i] += direction
                    grad[j] -= direction
                # Walls
                if current_centres[i, 0] < 0:
                    grad[i, 0] += 1.0
                else:
                    grad[i, 0] -= 1.0
                if current_centres[i, 1] < 0:
                    grad[i, 1] += 1.0
                else:
                    grad[i, 1] -= 1.0
        
        # Line search: try moving in gradient direction
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            continue
        grad /= grad_norm
        
        # Try several step sizes
        best_step_r = current_r
        best_step_centres = current_centres
        
        for step_mult in [0.01, 0.02, 0.05, 0.1, 0.2]:
            step = step_mult * step_size
            new_centres = current_centres + step * grad
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            new_r = _compute_radius(new_centres)
            
            if new_r > best_step_r:
                best_step_r = new_r
                best_step_centres = new_centres
        
        # Also try random perturbations mixed with gradient
        if np.random.rand() < 0.3:
            angle = np.random.uniform(0, 2 * np.pi)
            random_delta = step_size * 0.5 * np.array([np.cos(angle), np.sin(angle)])
            new_centres = current_centres + random_delta
            
            if not (np.any(new_centres < -0.5) or np.any(new_centres > 0.5)):
                new_r = _compute_radius(new_centres)
                if new_r > best_step_r:
                    best_step_r = new_r
                    best_step_centres = new_centres
        
        # Accept if improved
        if best_step_r > current_r:
            current_centres = best_step_centres
            current_r = best_step_r
            if current_r > best_r:
                best_centres = current_centres.copy()
                best_r = current_r
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Start with a good initial configuration
    # Try hexagonal packing
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # Local optimization
    if time.time() - start_time < TIME_LIMIT - 15:
        centres, current_r = _local_optimization(centres)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
# EVOLVE-BLOCK-END
