"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import KDTree

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
    # Minimum pairwise distance using cdist
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 22):
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                
                # Create the grid
                centres = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
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
                    best_centres = centres.copy()
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _violations_and_grad(centres, r):
    """
    Compute the sum of squared violations and its gradient.
    Violation for pair (i,j): max(0, 2r - dist_ij)^2
    Violation for wall i: max(0, r - dist_wall_i)^2
    """
    n = len(centres)
    # Compute pairwise distances
    # For N=200, 200x200 matrix is fine
    dists = np.linalg.norm(centres[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Pair violations
    # We only care about pairs where dist < 2r
    # To save time, we can mask, but for gradient we need the vectors
    # Let's compute full gradient terms
    # Term: (2r - d)^2 for d < 2r
    # dGrad = -2(2r-d) * (xi - xj)/d
    
    grad = np.zeros_like(centres)
    total_violation = 0.0
    
    # Vectorized pair calculation is tricky for gradient due to symmetry and masks.
    # Given N=200, a loop over i<j is 20k iterations, which is fast in Python/NumPy if vectorized per i.
    # But let's try to vectorize the mask.
    
    # Mask for active pairs
    active = dists < 2.0 * r
    # To avoid diagonal, set diagonal to False
    active[np.diag_indices(n)] = False
    
    # Calculate contributions
    # diff[i,j] = xi - xj
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    # d = dists
    # violation_term = max(0, 2r - d)^2
    # Only non-zero if active
    
    # We need sum over j for each i of: -2(2r - d_ij) * (xi - xj) / d_ij
    # Note: The gradient of (2r - d)^2 w.r.t xi is -2(2r-d) * (xi-xj)/d
    # Wait, d = ||xi - xj||. d/dxi ||xi-xj|| = (xi-xj)/||xi-xj||.
    # So d/dxi (2r - d)^2 = 2(2r-d) * (-1) * (xi-xj)/d = -2(2r-d) * (xi-xj)/d.
    
    # Let's compute the force matrix F where F[i] = sum_j grad_j
    # coeff[i,j] = -2(2r - d_ij) / d_ij for active pairs, 0 otherwise
    
    coeff = np.zeros_like(dists)
    # Avoid division by zero (diagonal is inf, so 1/inf is 0, but let's be safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Only compute where active
        coeff[active] = -2.0 * (2.0 * r - dists[active]) / dists[active]
    
    # grad[i] = sum_j coeff[i,j] * (xi - xj)
    # grad = coeff @ diff
    # Note: diff[i,j] = xi - xj
    grad_pairs = coeff @ diff
    
    total_violation_pairs = np.sum((2.0 * r - dists[active]) ** 2)
    
    # Wall violations
    # d_wall_i = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # This is non-smooth. We approximate by pushing away from the closest wall.
    # For gradient, we can treat it as a soft constraint or just use the subgradient.
    # Simpler: Add a penalty for being close to walls.
    # Let's use a smooth approximation or just handle it in the local polish.
    # For the main optimization, let's add a small penalty for exceeding bounds.
    # Actually, the problem requires centres INSIDE.
    # Let's add wall violations: max(0, r - d_wall)^2
    # d_wall_x_pos = 0.5 - x
    # d_wall_x_neg = 0.5 + x
    # d_wall_y_pos = 0.5 - y
    # d_wall_y_neg = 0.5 + y
    
    d_x_pos = 0.5 - centres[:, 0]
    d_x_neg = 0.5 + centres[:, 0]
    d_y_pos = 0.5 - centres[:, 1]
    d_y_neg = 0.5 + centres[:, 1]
    
    # Find min wall distance and which wall
    # We can just compute the gradient for each wall constraint separately and sum if active.
    # If d < r, it's active.
    
    # Gradient of (r - d_x_pos)^2 w.r.t x is -2(r - d_x_pos) * d/dx(d_x_pos) = -2(r - d_x_pos) * (-1) = 2(r - d_x_pos)
    # Wait, d_x_pos = 0.5 - x. d/dx = -1.
    # f = (r - (0.5 - x))^2 = (r - 0.5 + x)^2.
    # df/dx = 2(r - 0.5 + x) * 1.
    # This pushes x to increase (away from left wall). Correct.
    
    grad_walls = np.zeros_like(centres)
    total_violation_walls = 0.0
    
    # Left wall (x < -0.5 + r)
    active_l = d_x_neg < r  # d_x_neg is dist to left wall? No, d_x_neg = 0.5 + x. If x is -0.4, d=0.1.
    # Wait, x is in [-0.5, 0.5].
    # Left wall is x = -0.5. Distance is x - (-0.5) = x + 0.5.
    # So d_x_left = x + 0.5.
    # Right wall is x = 0.5. Distance is 0.5 - x.
    # Bottom wall is y = -0.5. Distance is y + 0.5.
    # Top wall is y = 0.5. Distance is 0.5 - y.
    
    d_left = centres[:, 0] + 0.5
    d_right = 0.5 - centres[:, 0]
    d_bottom = centres[:, 1] + 0.5
    d_top = 0.5 - centres[:, 1]
    
    # Left
    act_l = d_left < r
    if np.any(act_l):
        grad_walls[act_l, 0] += 2.0 * (r - d_left[act_l])  # Push right (+x)
        total_violation_walls += np.sum((r - d_left[act_l]) ** 2)
        
    # Right
    act_r = d_right < r
    if np.any(act_r):
        grad_walls[act_r, 0] += -2.0 * (r - d_right[act_r]) # Push left (-x)
        total_violation_walls += np.sum((r - d_right[act_r]) ** 2)
        
    # Bottom
    act_b = d_bottom < r
    if np.any(act_b):
        grad_walls[act_b, 1] += 2.0 * (r - d_bottom[act_b]) # Push up (+y)
        total_violation_walls += np.sum((r - d_bottom[act_b]) ** 2)
        
    # Top
    act_t = d_top < r
    if np.any(act_t):
        grad_walls[act_t, 1] += -2.0 * (r - d_top[act_t]) # Push down (-y)
        total_violation_walls += np.sum((r - d_top[act_t]) ** 2)
        
    grad = grad_pairs + grad_walls
    total_violation = total_violation_pairs + total_violation_walls
    
    return total_violation, grad


def _optimize_for_radius(centres, r, deadline):
    """
    Minimize violations for a fixed radius r.
    Returns optimized centres and final violation.
    """
    x0 = centres.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        v, _ = _violations_and_grad(c, r)
        return v
    
    def gradient(x):
        c = x.reshape(-1, 2)
        _, g = _violations_and_grad(c, r)
        return g.flatten()
    
    # Use L-BFGS-B
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 2), res.fun
    except Exception:
        return centres, np.inf


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3  # Leave 3s safety margin
    
    # 1. Initial Hexagonal Packing
    best_centres = _hexagonal_packing()
    current_r = _compute_radius(best_centres)
    
    # 2. Inflation Loop
    # We try to increase r step by step.
    # Start from current_r, try to go up to a theoretical max.
    # Theoretical max for 200 circles is around 0.0366.
    # Let's step by 0.001 initially, then refine.
    
    r_target = current_r
    step_size = 0.001
    max_r = 0.04
    
    # First, optimize the initial configuration for its current radius to clean it up
    best_centres, _ = _optimize_for_radius(best_centres, current_r, deadline)
    current_r = _compute_radius(best_centres)
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Try to increase radius
        r_trial = r_target + step_size
        if r_trial > max_r:
            break
            
        # Optimize for this new radius
        new_centres, viol = _optimize_for_radius(best_centres, r_trial, deadline)
        
        # Check if it's valid (violations near 0)
        if viol < 1e-8:
            # Success, update best
            best_centres = new_centres
            r_target = r_trial
            current_r = _compute_radius(best_centres)
            # Reduce step size slightly as we get closer to limit?
            # Or keep it constant.
            step_size = 0.0005
        else:
            # Failed to find a valid configuration for r_trial
            # Try a smaller step
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    # 3. Final Polish
    # Use a very fine step search on the radius to squeeze out more
    if time.time() < deadline - 5:
        # Binary search for the max r where violations can be minimized to ~0
        # Or just try a few small increments
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            r_trial = r_target + 0.0001
            new_centres, viol = _optimize_for_radius(best_centres, r_trial, deadline)
            if viol < 1e-8:
                best_centres = new_centres
                r_target = r_trial
            else:
                break

    final_r = _compute_radius(best_centres)
    
    # Fallback if something went wrong
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centres
