"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        n = len(centers)
        min_pair_dist = np.inf
        for i in range(n):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                d = np.min(dists)
                if d < min_pair_dist:
                    min_pair_dist = d
        if not np.isfinite(min_pair_dist):
            return 0.0
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)

    def generate_hex_target(n_rows, n_cols_per_row):
        """Generate a hexagonal lattice target inside the unit square."""
        pts = []
        if n_rows <= 0:
            return np.array([])
        
        # Calculate spacing to fit n_rows and n_cols
        # Hex packing: vertical spacing is sqrt(3)/2 * d, horizontal is d
        # Rows are offset by d/2
        # Total height: (n_rows - 1) * sqrt(3)/2 * d
        # Total width: n_cols * d (approx, depends on offset)
        
        d_max_x = 1.0 / n_cols_per_row
        d_max_h = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1)
        d = min(d_max_x, d_max_h)
        
        x_offset = -0.5 + d / 2
        y_offset = -0.5 + d / 2
        
        for row in range(n_rows):
            y = y_offset + row * d * math.sqrt(3) / 2
            if row % 2 == 0:
                n_in_row = n_cols_per_row
                x_start = x_offset
            else:
                n_in_row = n_cols_per_row - 1
                x_start = x_offset + d / 2
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with random points near edges to match N
            while len(pts) < N:
                x = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                y = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                pts = np.vstack([pts, [[x, y]]])
        
        return pts

    def ot_objective_and_grad(x, target_pts, beta=100.0):
        """
        Objective: Minimize negative radius (soft-min) + OT cost to target.
        The OT cost acts as a regularizer to maintain hexagonal structure.
        """
        c = x.reshape(-1, 2)
        
        # 1. Radius term (Soft-min)
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        min_pair = np.inf
        for i in range(N):
            diffs = c[i+1:] - c[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                d = np.min(dists)
                if d < min_pair:
                    min_pair = d
        
        if not np.isfinite(min_pair):
            return 1.0, np.zeros_like(x)
        
        r = min(min_wall, min_pair / 2.0)
        
        # Gradient of -r (approximate, focusing on the binding constraints)
        grad_r = np.zeros_like(c)
        
        # Wall gradient
        if min_wall < min_pair / 2.0:
            # Binding wall constraint
            mask_x = (np.abs(c[:,0]) > np.abs(c[:,1])) # Rough check
            # Actually, let's just push away from the closest wall
            # dist to left: 0.5 - c[:,0], right: 0.5 + c[:,0]
            # dist to bottom: 0.5 - c[:,1], top: 0.5 + c[:,1]
            d_left = 0.5 - c[:,0]
            d_right = 0.5 + c[:,0]
            d_bottom = 0.5 - c[:,1]
            d_top = 0.5 + c[:,1]
            
            min_d = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=1)
            # Find which wall is closest for each point? No, min_wall is global min.
            # We want to move points that are close to walls.
            # For simplicity, apply a force away from the nearest wall for all points
            # d_left < d_right -> move right (+x)
            grad_r[:, 0] += np.where(d_left < d_right, 1, -1)
            # d_bottom < d_top -> move up (+y)
            grad_r[:, 1] += np.where(d_bottom < d_top, 1, -1)
            
        # Pair gradient
        else:
            # Find closest pair
            min_d_global = min_pair
            # We need the indices of the closest pair
            # This is O(N^2) again, but we are in the objective function which is called often.
            # Optimization: only compute gradient for the closest few pairs?
            # Or just use a soft-min gradient.
            # Let's use a simpler heuristic: for each point, sum repulsive forces from neighbors within 1.5 * min_pair
            thresh = min_pair * 1.5
            for i in range(N):
                diffs = c - c[i]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                mask = (dists > 1e-8) & (dists < thresh)
                if np.any(mask):
                    # Force is proportional to 1/d^2 * direction
                    # Normalized direction
                    dirs = diffs[mask] / dists[mask, None]
                    weights = 1.0 / (dists[mask] ** 2)
                    grad_r[i] -= np.sum(dirs * weights[:, None], axis=0)
                    # Newton's third law
                    for j in np.where(mask)[0]:
                        if j != i:
                            grad_r[j] += np.sum(dirs * weights[:, None], axis=0) # Wait, this is wrong loop logic
                            
            # Correct Newton's third law implementation
            grad_r = np.zeros_like(c)
            for i in range(N):
                diffs = c - c[i]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                mask = (dists > 1e-8) & (dists < thresh)
                if np.any(mask):
                    idxs = np.where(mask)[0]
                    dirs = diffs[idxs] / dists[idxs, None]
                    weights = 1.0 / (dists[idxs] ** 2)
                    force_i = -np.sum(dirs * weights[:, None], axis=0)
                    grad_r[i] += force_i
                    for k, j in enumerate(idxs):
                        if j != i:
                            grad_r[j] -= force_i # Wait, force on j is opposite to force on i?
                            # Force on i is sum of (c_j - c_i)/|c_j-c_i|^2 * (-1)?
                            # Potential V = 1/r. F = -grad V = -(-1/r^2 * r_hat) = 1/r^2 * r_hat.
                            # r_hat points from i to j. So force on i is towards j? No, repulsive.
                            # Vector c_j - c_i points i->j. Force on i should be opposite (away from j).
                            # So force on i is - (c_j - c_i) / |c_j-c_i|^2.
                            # My code: diffs = c - c[i] -> c_j - c_i.
                            # force_i = - sum( dirs * weights ). Correct.
                            # Force on j is + sum( dirs * weights ).
                            grad_r[j] -= force_i # This adds the force accumulated from i to j?
                            # No, this loop is inefficient and slightly wrong in accumulation.
                            # Let's just compute pairwise forces properly.
        
        # Reset and compute forces properly
        grad_r = np.zeros_like(c)
        if min_pair < min_wall:
            thresh = min_pair * 2.0
            for i in range(N):
                diffs = c - c[i]
                dists_sq = np.sum(diffs**2, axis=1)
                dists = np.sqrt(dists_sq)
                mask = (dists > 1e-8) & (dists < thresh)
                if np.any(mask):
                    idxs = np.where(mask)[0]
                    # Force magnitude: 1/d^2
                    # Direction: away from neighbor
                    # f_ij = (c_i - c_j) / |c_i - c_j|^3 * k
                    # Here k=1 for simplicity
                    dirs = (c[i] - c[idxs]) / dists[idxs, None]
                    mags = 1.0 / (dists[idxs] ** 2)
                    force_i = np.sum(dirs * mags[:, None], axis=0)
                    grad_r[i] += force_i
                    for k, j in enumerate(idxs):
                        grad_r[j] -= force_i
        
        # 2. OT Cost term
        # Earth Mover's Distance approximation
        # Cost = sum |c_i - t_i|^2 / 2
        # We assume a fixed assignment (permutation) from initial state.
        # To make it differentiable and stable, we use the initial assignment.
        # This requires passing the assignment or using a soft assignment.
        # For simplicity in this single-function constraint, we assume the input x is ordered
        # similarly to target_pts. This is a strong assumption but works if we start close.
        # Better: use a soft assignment via Hungarian? Too slow.
        # Let's use a simple sum of squared distances to the nearest target point.
        # This is non-differentiable at boundaries but we can smooth it or just use it as a heuristic.
        # Actually, let's use the L-BFGS-B on the combined objective:
        # f(x) = -r(x) + lambda * OT_cost(x, target)
        
        # Simple OT cost: sum of min distance to target
        ot_cost = 0.0
        grad_ot = np.zeros_like(c)
        for i in range(N):
            dists_to_target = np.sqrt(np.sum((target_pts - c[i])**2, axis=1))
            min_idx = np.argmin(dists_to_target)
            min_dist = dists_to_target[min_idx]
            ot_cost += min_dist**2
            # Gradient: 2 * (c_i - t_min)
            grad_ot[i] += 2 * (c[i] - target_pts[min_idx])
            
        lambda_ot = 0.1
        total_obj = -r + lambda_ot * ot_cost
        total_grad = -grad_r + lambda_ot * grad_ot
        
        return total_obj, total_grad.flatten()

    def l_bfgs_polish(centers, target_pts, max_iter=500):
        """Polish using L-BFGS-B on the OT-regularized objective."""
        centers = centers.copy()
        x0 = centers.flatten()
        
        try:
            result = minimize(ot_objective_and_grad, x0, args=(target_pts, 100.0), method='L-BFGS-B',
                            bounds=[(-0.5, 0.5)] * (2 * N),
                            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
            new_centers = result.x.reshape(-1, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            return new_centers
        except Exception:
            return centers

    def coordinate_ascent_polish(centers, max_iter=20):
        """Final polish with coordinate ascent on pure radius."""
        centers = centers.copy()
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                orig = centers[i].copy()
                best_pos = orig
                best_r = compute_radius(centers)
                
                # Try moving in small steps in 8 directions
                for dx in [-0.0005, 0, 0.0005]:
                    for dy in [-0.0005, 0, 0.0005]:
                        if dx == 0 and dy == 0:
                            continue
                        new_center = orig + np.array([dx, dy])
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                        centers[i] = new_center
                        r = compute_radius(centers)
                        if r > best_r:
                            best_r = r
                            best_pos = new_center.copy()
                
                if not np.allclose(best_pos, orig):
                    centers[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        return centers

    # Main strategy:
    # 1. Generate hexagonal target
    # 2. Initialize centers near target
    # 3. Apply L-BFGS-B with OT regularization
    # 4. Final coordinate ascent

    best_centers = None
    best_radius = 0.0

    # Try different hexagonal configurations
    for n_rows in range(12, 18):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        
        target_pts = generate_hex_target(n_rows, n_cols)
        if len(target_pts) != N:
            continue
        
        # Initialize with target points (shuffled slightly to avoid symmetry locks?)
        # Actually, start exactly at target to help OT assignment
        centers = target_pts.copy()
        
        # Add small noise to break symmetry
        centers += np.random.normal(0, 0.001, centers.shape)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optimize
        if time.time() < deadline - 30:
            polished = l_bfgs_polish(centers, target_pts, max_iter=300)
            r = compute_radius(polished)
            
            if r > best_radius:
                best_radius = r
                best_centers = polished.copy()

    # Fallback to simple SA if OT fails or is slow
    if best_centers is None or best_radius < 0.035:
        # Simple hex grid + SA
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        grid = grid[:N]
        
        # Quick SA
        centers = grid.copy()
        T = 0.01
        for it in range(2000):
            if time.time() > deadline - 10:
                break
            i = np.random.randint(N)
            orig = centers[i].copy()
            centers[i] = orig + np.random.normal(0, T*0.5, 2)
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            r_new = compute_radius(centers)
            r_old = compute_radius(orig) # This is expensive, need to store current r
            # Approximation: accept if better, else with prob
            # For simplicity, just accept if better
            if r_new > best_radius:
                best_radius = r_new
                best_centers = centers.copy()
            T *= 0.999
        
        if best_centers is None:
            best_centers = grid

    # Final polish
    if best_centers is not None and time.time() < deadline - 5:
        polished = coordinate_ascent_polish(best_centers, max_iter=10)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    return best_centers
# EVOLVE-BLOCK-END
