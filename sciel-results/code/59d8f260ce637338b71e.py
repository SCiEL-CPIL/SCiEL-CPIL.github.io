"""
Strategy: Vectorized Force-Directed Local Search with Newton-Raphson Refinement.

This approach fundamentally differs from the previous SLSQP/Active-Set method by:
1. Abandoning general-purpose constrained optimizers (SLSQP) which are slow and prone to
   failure in high-dimensional non-convex spaces with many constraints.
2. Using a direct "Physics-based" local search:
   - Compute all pairwise distances.
   - Identify "violating" or "tight" pairs (dist < 2r + margin).
   - Apply repulsive forces proportional to the violation.
   - Project centers onto the feasible box (walls).
   - Iterate until convergence.
3. This is extremely fast because it is fully vectorized with NumPy (O(N^2) vector ops, 
   not Python loops).
4. Final polish: Use Newton-Raphson on the k-tightest constraints to maximize r 
   locally. This is a "linearization" step that is very precise for the final few digits.

The key insight is that for packing problems, the local landscape is dominated by the 
"contact graph" (pairs at distance 2r). By directly manipulating these pairs with forces, 
we can escape local minima more effectively than gradient-based methods that get stuck 
on the non-smooth min function.
"""

import numpy as np
import time

N = 50
HALF = 0.5
DEADLINE = 140.0  # Safety margin for 150s limit

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    
    # Wall distances
    # Distance to nearest wall for each circle
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    # dist to left wall: x - (-0.5) = x + 0.5
    # dist to right wall: 0.5 - x
    # dist to bottom wall: y + 0.5
    # dist to top wall: 0.5 - y
    # min wall dist = min(x+0.5, 0.5-x, y+0.5, 0.5-y) = 0.5 - max(|x|, |y|)
    wall_dists = HALF - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_dist, min_wall_dist)
    return max(0.0, r)

def _hex_init():
    """Generate a high-quality hexagonal packing initialization."""
    # Best known r is ~0.071377. Hexagonal packing has density pi/(2*sqrt(3)) ~ 0.9069.
    # For N=50, area = 1. 50 * pi * r^2 <= 1 * 0.9069 => r <= sqrt(0.9069 / (50*pi)) ~ 0.0758.
    # So 0.075 is a good starting guess.
    r_guess = 0.075
    dx = 2 * r_guess
    dy = r_guess * np.sqrt(3)
    
    points = []
    # Generate a grid larger than the square to ensure we have enough points
    range_val = 1.2
    nx = int(range_val / dx) + 2
    ny = int(range_val / dy) + 2
    
    for i in range(-ny, ny):
        for j in range(-nx, nx):
            x = j * dx
            y = i * dy
            if i % 2 != 0:
                x += dx / 2
            points.append([x, y])
            
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    # Select the N points closest to the center
    dists_to_center = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    points = points[indices]
    
    # Scale to fit inside the square with some margin
    max_ext = np.max(np.abs(points))
    target_max = HALF - r_guess * 0.9  # Leave a small margin
    if max_ext > target_max:
        scale = target_max / max_ext
        points *= scale
        
    return points

def _force_directed_step(centers, r_target, steps=500, step_size=0.5, tol=1e-8):
    """
    Vectorized force-directed local search.
    We try to maintain a radius r_target.
    If circles are too close, push them apart.
    If circles are too close to walls, push them in.
    
    This is effectively a gradient ascent on the minimum distance, but using 
    explicit forces from the "violating" pairs.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    for _ in range(steps):
        # 1. Compute pairwise differences and distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # 2. Identify violations: dist < 2 * r_target
        # We want to increase the minimum distance. 
        # If we fix r_target, we push pairs with dist < 2*r_target apart.
        # However, to actually increase r, we should push pairs that are currently 
        # at the minimum distance (or close to it) apart.
        
        # Let's use a "soft" constraint: 
        # Force on i from j is proportional to (2*r_target - dist_ij) if dist_ij < 2*r_target.
        # This pushes them apart until they are at 2*r_target.
        
        # But we also want to increase r_target over time.
        # Strategy: 
        # - Maintain a "desired" radius r_desired.
        # - Push pairs with dist < 2*r_desired apart.
        # - Push circles with wall_dist < r_desired away from walls.
        # - If no violations, increase r_desired slightly.
        
        # For simplicity and robustness, let's just maximize the current minimum distance.
        # Identify the pairs that are "tight" (dist < 2*current_r + margin).
        # Actually, the standard "Repulsion" method is:
        # For all pairs, force = k * (d_min - dist) if dist < d_min, else 0.
        # Here d_min is the target separation.
        
        # Let's use the current minimum distance as the target separation.
        # min_dist = np.min(dists)
        # target_sep = min_dist
        # This would just maintain the current state.
        
        # Better: Use a target slightly larger than the current minimum to force expansion.
        # But we must respect walls.
        
        # Let's use the "Projection" method:
        # 1. Compute forces from all pairs where dist < 2 * r_current.
        #    Actually, we want to increase the min dist. 
        #    So we identify pairs with dist < 2 * r_current + delta.
        #    Let's just use all pairs where dist < 2 * r_current.
        
        # A more robust approach for packing:
        # 1. Compute all dists.
        # 2. Find min_dist.
        # 3. Identify pairs with dist < min_dist + eps.
        # 4. Push these pairs apart.
        # 5. Identify circles with wall_dist < r_current.
        # 6. Push them away from walls.
        
        # Let's implement a generic "Repulsion" step.
        # We define a "soft" radius r_soft = current_r * 1.001.
        # We push pairs with dist < 2*r_soft apart.
        # We push circles with wall_dist < r_soft away from walls.
        
        r_soft = current_r * 1.001 + 1e-6
        
        # Pair forces
        # Force on i from j: (c_i - c_j) / dist * (2*r_soft - dist)
        # This is the gradient of the penalty (max(0, 2*r_soft - dist))^2 / 2
        
        # Vectorized force calculation
        # We need to be careful with dist=0.
        # diff is (N, N, 2)
        # dists is (N, N)
        
        # Mask for pairs to push
        mask = dists < 2 * r_soft
        np.fill_diagonal(mask, False)
        
        if not np.any(mask):
            # No pair violations. Check walls.
            wall_dists = HALF - np.max(np.abs(centers), axis=1)
            wall_mask = wall_dists < r_soft
            
            if not np.any(wall_mask):
                # No violations at all. Increase r_soft? 
                # Actually, if no violations, we can just break and let the caller increase r.
                # But for now, let's just break.
                break
            
            # Wall forces
            # For each circle i with wall_mask[i]:
            # If x < -HALF + r_soft, push right (+x)
            # If x > HALF - r_soft, push left (-x)
            # If y < -HALF + r_soft, push up (+y)
            # If y > HALF - r_soft, push down (-y)
            
            forces = np.zeros_like(centers)
            for i in range(N):
                if wall_mask[i]:
                    x, y = centers[i]
                    if x < -HALF + r_soft:
                        forces[i, 0] += (r_soft - (x + HALF))
                    if x > HALF - r_soft:
                        forces[i, 0] -= (r_soft - (HALF - x))
                    if y < -HALF + r_soft:
                        forces[i, 1] += (r_soft - (y + HALF))
                    if y > HALF - r_soft:
                        forces[i, 1] -= (r_soft - (HALF - y))
            
            centers += step_size * forces
            # Project onto box
            centers = np.clip(centers, -HALF + r_soft, HALF - r_soft)
            # Update current_r
            new_r = _compute_radius(centers)
            if new_r < current_r - 1e-9:
                # We made it worse. Revert? 
                # For now, just accept and hope next step fixes it.
                # Or better: keep the old centers if they were better.
                pass
            current_r = new_r
            continue
        
        # Calculate pair forces
        # We want to move c_i away from c_j.
        # Force on i = sum over j of (c_i - c_j) * (2*r_soft - dist_ij) / dist_ij
        # Note: (c_i - c_j) / dist_ij is the unit vector from j to i.
        
        # To vectorize:
        # unit_vecs = diff / dists[:, :, None]
        # weights = (2*r_soft - dists) * mask
        # forces = sum(weights[:, :, None] * unit_vecs, axis=1)
        
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, None]
        
        # Weights
        weights = (2 * r_soft - dists) * mask
        # Weights is (N, N)
        
        # Sum forces over j
        forces = np.sum(weights[:, :, None] * unit_vecs, axis=1)
        
        # Update centers
        centers += step_size * forces
        
        # Project onto box (hard constraint)
        # We must ensure circles stay inside [-HALF+r, HALF-r]?
        # No, the radius is variable. We just need centers in [-HALF, HALF].
        # But if we push a circle against the wall, it might violate the "inside" constraint 
        # if we consider the circle body. 
        # The constraint is: center must be in [-HALF+r, HALF-r] for a fixed r.
        # But r is changing.
        # Safest: Clip centers to [-HALF, HALF].
        # Then compute new r.
        centers = np.clip(centers, -HALF, HALF)
        
        new_r = _compute_radius(centers)
        if new_r > current_r:
            current_r = new_r
        else:
            # If we got worse, we might need to dampen the step or revert.
            # For simplicity, we just continue. The forces should correct it.
            pass
            
        if current_r > r_target and abs(current_r - r_target) < tol:
             break

    return centers

def _newton_refine(centers, r_current, max_iters=50):
    """
    Newton-Raphson refinement on the tightest constraints.
    This is a "Local Search" step that solves the linearized problem:
    Maximize r
    Subject to:
      ||c_i - c_j|| >= 2r  for tight pairs
      0.5 - |c_ik| >= r    for tight walls
    
    We linearize the constraints around the current point.
    Let d_i be the displacement of circle i.
    Let dr be the change in radius.
    
    Pair constraint (i, j):
    ||c_i + d_i - (c_j + d_j)|| >= 2(r + dr)
    Approx: (c_i - c_j) . (d_i - d_j) / dist_ij >= 2 dr
    
    Wall constraint (i, wall k):
    0.5 - (c_ik + d_ik) >= r + dr  (assuming c_ik > 0, right wall)
    -d_ik >= dr
    
    We want to maximize dr subject to these linear inequalities.
    This is a Linear Program (LP).
    We can solve this LP using scipy.optimize.linprog.
    """
    centers = centers.copy()
    r = r_current
    
    for _ in range(max_iters):
        # Identify tight constraints
        # Pairs: dist < 2r + eps
        # Walls: wall_dist < r + eps
        eps = 1e-5
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        pair_mask = dists < 2 * r + eps
        np.fill_diagonal(pair_mask, False)
        
        wall_dists = HALF - np.max(np.abs(centers), axis=1)
        # We need to know WHICH wall is tight for each circle.
        # For a circle i, the tight wall is the one minimizing 0.5 - |c_ik|.
        # Actually, the constraint is min(0.5-|x|, 0.5-|y|) >= r.
        # If 0.5-|x| < r+eps, then x is tight.
        # If 0.5-|y| < r+eps, then y is tight.
        
        # Let's identify active walls.
        active_walls = [] # list of (i, axis, sign)
        # sign: +1 for right/top (constraint: -x - dr >= 0 => -dx >= dr)
        #        -1 for left/bottom (constraint: x + dr >= 0 => dx >= dr)
        
        for i in range(N):
            x, y = centers[i]
            # Right wall: 0.5 - x
            if 0.5 - x < r + eps:
                active_walls.append((i, 0, -1)) # -dx >= dr
            # Left wall: 0.5 + x
            if 0.5 + x < r + eps:
                active_walls.append((i, 0, 1))  # dx >= dr
            # Top wall: 0.5 - y
            if 0.5 - y < r + eps:
                active_walls.append((i, 1, -1)) # -dy >= dr
            # Bottom wall: 0.5 + y
            if 0.5 + y < r + eps:
                active_walls.append((i, 1, 1))  # dy >= dr
        
        # If no active constraints, we can just increase r?
        # No, if no active constraints, the packing is loose.
        # We should find the next closest pair/wall.
        # But for Newton, if no constraints, the LP is unbounded.
        # So we must have at least one constraint.
        # If none, break.
        if not np.any(pair_mask) and not active_walls:
            break
        
        # Variables: d_1x, d_1y, ..., d_Nx, d_Ny, dr
        # Total vars: 2N + 1
        n_vars = 2 * N + 1
        
        # Objective: Maximize dr => Minimize -dr
        # c = [0, 0, ..., 0, -1]
        c_obj = np.zeros(n_vars)
        c_obj[-1] = -1
        
        # Constraints: A_ub @ vars <= b_ub
        # Pair (i, j):
        # (c_i - c_j) . (d_i - d_j) / dist_ij >= 2 dr
        # => - (c_i - c_j) . (d_i - d_j) / dist_ij + 2 dr <= 0
        
        A_rows = []
        b_vals = []
        
        # Pair constraints
        i_indices, j_indices = np.where(pair_mask)
        for i, j in zip(i_indices, j_indices):
            if i >= j: continue # Only i < j to avoid duplicates? 
            # Actually, the mask is symmetric. We should only take upper triangle.
            # Let's just iterate i < j.
            
            # Wait, np.where returns all i, j.
            # Let's filter i < j.
            
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = dists[i, j]
            
            if d < 1e-9: continue
            
            # Unit vector from j to i
            ux = dx / d
            uy = dy / d
            
            # Constraint: ux * (d_ix - d_jx) + uy * (d_iy - d_jy) >= 2 dr
            # => -ux * d_ix + ux * d_jx - uy * d_iy + uy * d_jy + 2 dr <= 0
            
            row = np.zeros(n_vars)
            row[2*i] = -ux
            row[2*i+1] = -uy
            row[2*j] = ux
            row[2*j+1] = uy
            row[-1] = 2.0
            
            A_rows.append(row)
            b_vals.append(0.0)
            
        # Wall constraints
        # Right (i, 0, -1): -dx >= dr => -dx - dr <= 0
        # Left (i, 0, 1): dx >= dr => -dx + dr <= 0  (Wait: dx >= dr => -dx <= -dr)
        # Let's be careful.
        # Constraint: sign * d_ik >= dr  =>  -sign * d_ik - dr <= 0
        
        for i, axis, sign in active_walls:
            row = np.zeros(n_vars)
            # d_ik is at index 2*i + axis
            row[2*i + axis] = -sign
            row[-1] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
        
        if not A_rows:
            break
            
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        
        # Bounds: 
        # dr can be anything (but we expect positive).
        # Displacements should be small.
        # Let's bound dr to [0, 0.01] to ensure monotonic improvement.
        # And displacements to [-0.01, 0.01].
        
        bounds = [(-0.01, 0.01)] * (2*N) + [(0, 0.01)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                dr = res.x[-1]
                if dr > 1e-8:
                    d_centers = res.x[:2*N].reshape(N, 2)
                    centers += d_centers
                    r += dr
                else:
                    break
            else:
                break
        except Exception:
            break
            
    return centers, r

def construct():
    start_time = time.time()
    
    # 1. Initialization
    centers = _hex_init()
    current_r = _compute_radius(centers)
    
    best_centers = centers.copy()
    best_r = current_r
    
    # 2. Force-Directed Local Search
    # We run multiple rounds of force-directed search with increasing target radii.
    # Or just run it and see if r increases.
    
    # To escape local minima, we can add small random noise periodically.
    
    max_rounds = 20
    for round_num in range(max_rounds):
        if time.time() > DEADLINE:
            break
            
        # Force directed step
        # We don't have a fixed target r, we just want to increase the min dist.
        # The _force_directed_step function uses current_r * 1.001 as target.
        # This should slowly push the packing tighter.
        
        new_centers = _force_directed_step(centers, current_r, steps=200, step_size=0.1)
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers
            current_r = new_r
        else:
            # If no improvement, add a kick to escape local minimum.
            # The kick should be small.
            kick = np.random.normal(0, 0.001, (N, 2))
            centers = centers + kick
            centers = np.clip(centers, -HALF, HALF)
            current_r = _compute_radius(centers)
            
    # 3. Newton-Raphson Refinement
    if time.time() < DEADLINE - 10:
        refined_centers, refined_r = _newton_refine(best_centers, best_r, max_iters=50)
        if refined_r > best_r:
            best_r = refined_r
            best_centers = refined_centers
        else:
            # Try Newton from the current "centers" state too, in case it's better
            refined_centers2, refined_r2 = _newton_refine(centers, current_r, max_iters=50)
            if refined_r2 > best_r:
                best_r = refined_r2
                best_centers = refined_centers2

    return best_centers

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the EVOLVE-BLOCK as per the instructions?
# "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
# The instructions say: "Output the COMPLETE program in a single
#
