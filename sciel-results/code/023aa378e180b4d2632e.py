"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _radius(pts):
    """Compute the true maximum radius for a given set of centers."""
    if len(pts) == 0:
        return 0.0
    # Distance to walls
    dx = np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5)
    dy = np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    r_wall = np.min(np.minimum(dx, dy))
    
    if r_wall < 0:
        return 0.0 # Invalid if outside

    # Distance to other points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)

def _make_hexagonal_grid(n):
    """Generate a hexagonal grid of n points centered at origin."""
    # For n=50, a grid of roughly 7x8 or 8x7 is good.
    # Hex packing density ~0.9069
    # Area per circle ~ pi*r^2. Total area ~ n*pi*r^2.
    # Square area 1. r ~ sqrt(1/(n*pi/sqrt(3))) ~ sqrt(sqrt(3)/(n*pi))
    # For n=50, r ~ sqrt(1.732 / 157) ~ sqrt(0.011) ~ 0.105 (ideal, no boundary loss)
    # Actual r is smaller due to boundary.
    
    pts = []
    # Try a range of grid sizes and pick the one that fits best or just use a fixed good one
    # Let's generate a large hex grid and select the n closest to center
    # Spacing s. Distance between centers is s.
    # We want to fit in [-0.5, 0.5].
    
    # Estimate rows/cols
    # n ~ rows * cols
    # Let's try rows=7, cols=8 -> 56 points.
    rows = 7
    cols = 8
    
    # Calculate optimal spacing to fit in box
    # Width = (cols - 0.5) * s  (approx, due to stagger)
    # Height = (rows - 1) * s * sqrt(3)/2 + s (approx)
    # Actually, for hex grid:
    # x range: c in [0, cols-1], x = c*s. Width = (cols-1)*s. Plus stagger s/2.
    # Let's just compute positions for a specific s and check bounds.
    
    # We want to maximize s.
    # Let's iterate s from large to small? Or just calculate based on max extent.
    # Max x extent: (cols-1)*s + s/2 (if staggered)
    # Max y extent: (rows-1)*s*sqrt(3)/2
    
    # Let's just generate with s=1 and scale to fit.
    raw_pts = []
    for r in range(rows):
        for c in range(cols):
            x = c * 1.0
            y = r * math.sqrt(3) / 2
            if r % 2 == 1:
                x += 0.5
            raw_pts.append([x, y])
    raw_pts = np.array(raw_pts)
    
    # Center it
    raw_pts[:, 0] -= raw_pts[:, 0].mean()
    raw_pts[:, 1] -= raw_pts[:, 1].mean()
    
    # Scale to fit in [-0.5, 0.5] with a small margin
    max_extent = np.max(np.abs(raw_pts))
    # Leave some margin for optimization to push outwards
    scale = 0.45 / max_extent 
    pts = raw_pts * scale
    
    # Select n closest to origin? Or just first n?
    # Selecting closest to origin might create a dense center and empty corners, which is bad for wall constraints.
    # Better to just take the first n from a generation that fills the space reasonably.
    # Since we generated rows*cols, if rows*cols >= n, we can just take the first n?
    # No, that might be biased to one corner of the grid index.
    # Let's sort by distance to center and take n? 
    # Actually, for packing, we want uniform distribution.
    # If we generated exactly n points in a grid shape, it's best.
    # 7*8=56. Let's remove the 6 furthest from center.
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    # Keep the n closest to center to ensure the bulk is central, but this might leave corners empty.
    # Alternative: Keep the ones that form the most compact set?
    # Let's just take the first N from a well-ordered grid.
    # Re-generate in a way that fills the square better?
    # For now, take n closest to center is a safe bet for initialization, 
    # then the expansion phase will push them to the walls.
    
    return pts[indices[:n]]

def _expand_radius(pts, max_iter=1000, dt=0.0001):
    """
    Hard-disk expansion: increase radius, resolve collisions by pushing points apart.
    This is essentially a soft repulsion force approach but with a growing 'hard' core.
    """
    n = len(pts)
    r = _radius(pts)
    current_r = r * 0.99 # Start slightly smaller to avoid immediate collisions
    
    for it in range(max_iter):
        # Try to increase radius
        current_r += dt
        
        # Check for collisions and resolve
        # We want to push points apart if they are closer than 2*current_r
        # And push points inward if they are closer to wall than current_r
        
        forces = np.zeros_like(pts)
        
        # Wall constraints
        # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
        # Soft penalty: force proportional to violation
        # But we want to maximize r, so we want to allow points to go to the wall.
        # Actually, in expansion, we fix r and move points to satisfy constraints.
        # If a point is too close to wall, push it away from wall.
        
        margin = 1e-6
        wall_r = current_r + margin
        
        # X walls
        # Right wall: x > 0.5 - wall_r -> push left (force_x < 0)
        viol_r = np.clip(pts[:, 0] - (0.5 - wall_r), 0, None)
        forces[:, 0] -= viol_r * 10.0
        
        # Left wall: x < -0.5 + wall_r -> push right (force_x > 0)
        viol_l = np.clip((-0.5 + wall_r) - pts[:, 0], 0, None)
        forces[:, 0] += viol_l * 10.0
        
        # Y walls
        viol_top = np.clip(pts[:, 1] - (0.5 - wall_r), 0, None)
        forces[:, 1] -= viol_top * 10.0
        
        viol_bot = np.clip((-0.5 + wall_r) - pts[:, 1], 0, None)
        forces[:, 1] += viol_bot * 10.0
        
        # Pairwise collisions
        # If dist < 2*current_r, push apart.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist = 2.0 * current_r
        # Violation: min_dist - dist. If positive, push apart.
        # Force is along diff vector.
        # F_ij = k * (min_dist - dist_ij) * unit_vector_ij
        
        # To avoid division by zero, add small epsilon to dist
        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        violations = np.clip(min_dist - dists, 0, None)
        # Force on i due to j is -unit_vec_ij * violation (push away from j)
        # Sum over j
        # F_i = sum_j ( - unit_vec_ij * violations_ij * k )
        # Note: unit_vec_ij points from j to i? 
        # diff = pts[i] - pts[j]. So unit_vec points from j to i.
        # If dist < min_dist, we want to move i further from j.
        # So force should be in direction of (pts[i] - pts[j]), i.e., unit_vec.
        # Wait. If I am at 0 and j is at 1. diff = -1. unit_vec = -1.
        # I want to move to -1. Force should be -1.
        # So F_i = sum_j ( unit_vec_ij * violations_ij * k )
        
        k_pair = 10.0
        forces += np.sum(unit_vecs * violations[:, :, np.newaxis] * k_pair, axis=1)
        
        # Apply forces
        pts += forces * 0.01 # Learning rate / step size
        
        # Project back to valid region? 
        # The forces should keep them inside, but numerical errors might push out.
        # Clip to a slightly larger box to prevent divergence, then let forces pull back.
        # Actually, if we clip to [-0.5, 0.5], we might stop them from finding the optimal wall position.
        # But the wall force is strong.
        pts = np.clip(pts, -0.5, 0.5)
        
        # Check if we made progress
        new_r = _radius(pts)
        if new_r < current_r - 1e-5:
            # We broke something, rollback?
            # For simplicity, just accept and continue. The forces should correct it.
            pass
            
        # Convergence check: if forces are small
        if np.max(np.abs(forces)) < 1e-4:
            break
            
    return pts, _radius(pts)

def _slsqp_polish(pts, max_iter=100):
    """
    Use SLSQP to maximize r subject to constraints.
    Variables: x (2N coords), r (1 scalar). Total 2N+1 = 101 vars.
    Objective: -r (minimize -r)
    Constraints:
    1. 2r - dist_ij <= 0 for all i<j
    2. r - (0.5 - x_i) <= 0
    3. r - (x_i + 0.5) <= 0
    4. r - (0.5 - y_i) <= 0
    5. r - (y_i + 0.5) <= 0
    """
    from scipy.optimize import minimize
    
    n = len(pts)
    x0 = np.concatenate([pts.flatten(), [_radius(pts)]])
    
    def objective(x):
        return -x[-1]
        
    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad
        
    constraints = []
    
    # Pairwise constraints: 2r - dist_ij <= 0
    # There are N*(N-1)/2 constraints. For N=50, 1225 constraints.
    # This might be too many for SLSQP?
    # SLSQP handles inequality constraints.
    # Let's see if it's feasible.
    
    # Wall constraints: 4 per point = 200 constraints.
    
    # Total ~1425 constraints.
    
    # Define constraints functions
    # To speed up, we can vectorize the evaluation inside the constraint function if possible,
    # but scipy minimize expects a single constraint function returning a scalar or array?
    # If it returns an array, it's treated as multiple constraints?
    # No, scipy.optimize.minimize constraints are dictionaries with 'type': 'ineq' and 'fun' returning scalar or array.
    # If 'fun' returns array, each element is a constraint?
    # Documentation says: "fun(x) >= 0". If fun returns array, all elements must be >= 0.
    
    # Let's group constraints to reduce overhead?
    # Or just define one big function for all pairwise?
    
    def pairwise_constraints(x):
        p = x[:2*n].reshape(n, 2)
        r = x[-1]
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Constraint: dists - 2*r >= 0
        # Return all upper triangle
        iu = np.triu_indices(n, k=1)
        return dists[iu] - 2.0 * r
        
    def wall_constraints(x):
        p = x[:2*n].reshape(n, 2)
        r = x[-1]
        # 0.5 - x >= r  => 0.5 - x - r >= 0
        # x + 0.5 >= r  => x + 0.5 - r >= 0
        # 0.5 - y >= r
        # y + 0.5 >= r
        c1 = 0.5 - p[:, 0] - r
        c2 = p[:, 0] + 0.5 - r
        c3 = 0.5 - p[:, 1] - r
        c4 = p[:, 1] + 0.5 - r
        return np.concatenate([c1, c2, c3, c4])
        
    constraints.append({'type': 'ineq', 'fun': pairwise_constraints})
    constraints.append({'type': 'ineq', 'fun': wall_constraints})
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * n) + [(0, 0.5)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                       bounds=bounds, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': 1e-8})
        
        if res.success or (res.fun < -x0[-1]): # If it improved
            # Extract best
            best_x = res.x
            best_pts = best_x[:2*n].reshape(n, 2)
            best_r = best_x[-1]
            
            # Verify
            true_r = _radius(best_pts)
            if true_r > _radius(pts):
                return best_pts, true_r
            else:
                return pts, _radius(pts)
        else:
            return pts, _radius(pts)
    except Exception:
        return pts, _radius(pts)

def construct():
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin
    
    # 1. Initialize
    pts = _make_hexagonal_grid(N)
    
    # 2. Expand Radius
    if time.time() - start_time < timeout - 20:
        pts, r = _expand_radius(pts, max_iter=500, dt=0.0002)
        
    # 3. Polish with SLSQP
    if time.time() - start_time < timeout - 5:
        pts, r = _slsqp_polish(pts, max_iter=50)
        
    # 4. Final local refinement if time permits
    # Maybe run SLSQP again with tighter tolerance?
    if time.time() - start_time < timeout - 2:
         pts, r = _slsqp_polish(pts, max_iter=20)
         
    # Ensure validity
    pts = np.clip(pts, -0.5, 0.5)
    
    return pts

# EVOLVE-BLOCK-END
