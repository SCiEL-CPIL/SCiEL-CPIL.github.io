"""
Force-Directed Layout (FDL) for Circle Packing.
Strategy:
1. Initialize with a hexagonal grid.
2. Use a deterministic Force-Directed Layout solver:
   - Repulsive forces between all pairs (inverse power law) to maximize separation.
   - Attractive forces to a target radius r_target to "squeeze" the configuration.
   - Wall repulsion to keep circles inside.
3. Iterate by increasing r_target (binary search style) to find the maximum feasible radius.
4. Final polish with L-BFGS-B on the violation loss for the best found radius.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = 0.5
        
    return min(min_wall, min_pair)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8, 9, 10]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r_est = min(r_horiz, r_vert)
        
        if r_est <= best_r:
            continue

        centers = []
        r = r_est
        y_spacing = r * np.sqrt(3.0)
        total_height = 2*r + (n_rows - 1) * y_spacing
        y_start = -0.5 + r + (1.0 - total_height) / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + (1.0 - x_span) / 2.0
            
            offset = y_spacing * 0.5 if row_idx % 2 == 1 else 0.0
            
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        centers = np.clip(centers, -0.5, 0.5)
        
        r_actual = _score(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _fdl_step(cents, r_target, dt=0.001, rep_power=4, rep_strength=1.0, attr_strength=10.0, wall_strength=50.0):
    """
    Perform one step of force-directed layout.
    Returns new centers and total force magnitude (for convergence check).
    """
    n = len(cents)
    forces = np.zeros_like(cents)
    
    # 1. Pairwise Repulsion
    # F_ij = k * (r_target / d_ij)^rep_power * (c_i - c_j) / d_ij
    # We want to push them apart if they are closer than 2*r_target? 
    # Actually, standard FDL for packing: Repulsion is always on, but stronger when close.
    # Let's use a soft repulsion: F = k * (1/d^2) * dir, but scaled by r_target to keep units consistent?
    # Better: F_rep = k * (r_target / d)^rep_power * dir / d
    # This creates a "desired" separation of r_target.
    
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e-6) # Avoid div by zero
    
    # Unit vectors
    # dists_inv = 1 / dists
    # dirs = diff / dists[:, :, np.newaxis]
    
    # Magnitude of repulsive force
    # If d < 2*r_target, we want strong repulsion.
    # Let's define force magnitude: k * (r_target / d)^rep_power
    # This is large when d is small.
    
    # To avoid numerical instability, we can mask d < epsilon
    # But with N=50, O(N^2) is fine.
    
    # Force on i from j:
    # F_ij = rep_strength * (r_target / d_ij)^rep_power * (c_i - c_j) / d_ij
    # Note: (c_i - c_j) / d_ij is the unit vector pointing from j to i.
    # So F_ij pushes i away from j.
    
    # Vectorized:
    # d_mat = dists
    # factor = (r_target / d_mat) ** rep_power / d_mat  # 1/d extra for unit vector scaling?
    # Let's be precise:
    # unit_vec = diff / dists[:,:,None]
    # mag = rep_strength * (r_target / dists) ** rep_power
    # forces += sum(mag[:,:,None] * unit_vec, axis=1)
    
    # Optimization:
    # inv_d = 1.0 / dists
    # unit_vec = diff * inv_d[:, :, np.newaxis]
    # mag = rep_strength * (r_target * inv_d) ** rep_power
    # force_contribution = mag[:, :, np.newaxis] * unit_vec
    # forces = np.sum(force_contribution, axis=1)
    
    # Let's do it step-by-step for clarity and memory safety if N grows, but N=50 is small.
    inv_d = 1.0 / dists
    unit_vec = diff * inv_d[:, :, np.newaxis]
    
    # Scale repulsion: We want the force to be strong enough to overcome attraction.
    # The "natural" length scale is r_target.
    # If we use (r_target / d)^p, the force is 1 when d=r_target.
    # We want it to be 0 at d=2*r_target (touching).
    # So we can use a cutoff or a smooth decay.
    # Let's use: mag = rep_strength * (r_target / d) ** rep_power * (1 - d/(2*r_target)) for d < 2*r_target, else 0.
    # This ensures no force when touching.
    
    mask = dists < 2.0 * r_target
    # To avoid division by zero in mask calculation if r_target is 0, but r_target > 0.
    
    mag = np.zeros_like(dists)
    # Calculate only where mask is true
    if np.any(mask):
        # r_target / d
        ratio = r_target * inv_d[mask]
        # (1 - d/(2*r_target)) = (1 - 0.5 * d/r_target) = (1 - 0.5 * inv_r_target * d)
        # Simpler: (1 - d / (2*r_target))
        decay = 1.0 - dists[mask] / (2.0 * r_target)
        # Ensure decay is positive
        decay = np.maximum(decay, 0.0)
        mag[mask] = rep_strength * (ratio ** rep_power) * decay
        
    force_contribution = mag[:, :, np.newaxis] * unit_vec
    forces += np.sum(force_contribution, axis=1)
    
    # 2. Attraction to Target Radius?
    # The FDL usually relies on a "box" or external potential.
    # Here, we want to maximize r. 
    # A common trick: Apply an attractive force towards the center? No, that clumps them.
    # Instead, we use the "target radius" as a constraint in a penalty function, 
    # OR we use a "spring" that pulls circles together to a distance of 2*r_target.
    # Wait, if we pull them together to 2*r_target, they will clump.
    # We need to balance: Repulsion pushes them apart, Attraction pulls them in.
    # Equilibrium is when Repulsion = Attraction.
    # If we increase r_target, the Repulsion range increases (since it depends on r_target),
    # effectively pushing them further apart to maintain equilibrium.
    
    # Let's add a weak attractive force to the center of mass to keep them compact?
    # Or better: Attractive force towards a "grid" or just a global center?
    # Actually, a pure repulsive force in a box will push them to the walls.
    # We need wall repulsion.
    
    # 3. Wall Repulsion
    # Distance to wall: d_wall = 0.5 - |x|, 0.5 - |y|
    # If d_wall < r_target, apply repulsive force pushing inwards.
    # F_wall = wall_strength * (r_target - d_wall) / r_target (normalized)
    
    # X walls
    d_left = 0.5 + cents[:, 0]   # dist to left wall
    d_right = 0.5 - cents[:, 0]  # dist to right wall
    
    # Force X
    f_x = np.zeros(n)
    # Left wall: if d_left < r_target, push right (+x)
    mask_l = d_left < r_target
    if np.any(mask_l):
        # Strength proportional to penetration
        pen = r_target - d_left[mask_l]
        f_x[mask_l] += wall_strength * pen
    # Right wall: if d_right < r_target, push left (-x)
    mask_r = d_right < r_target
    if np.any(mask_r):
        pen = r_target - d_right[mask_r]
        f_x[mask_r] -= wall_strength * pen
        
    # Y walls
    d_bottom = 0.5 + cents[:, 1]
    d_top = 0.5 - cents[:, 1]
    
    f_y = np.zeros(n)
    mask_b = d_bottom < r_target
    if np.any(mask_b):
        pen = r_target - d_bottom[mask_b]
        f_y[mask_b] += wall_strength * pen
    mask_t = d_top < r_target
    if np.any(mask_t):
        pen = r_target - d_top[mask_t]
        f_y[mask_t] -= wall_strength * pen
        
    forces[:, 0] += f_x
    forces[:, 1] += f_y
    
    # 4. Update positions
    # Damping to stabilize
    damping = 0.9
    # We don't have velocity in this simple version, so we just move by force * dt
    # This is like gradient descent on a potential.
    new_cents = cents + forces * dt
    new_cents = np.clip(new_cents, -0.5, 0.5)
    
    # Check for overlap violations (for monitoring)
    # We don't strictly enforce non-overlap in FDL, but the forces should prevent it.
    
    # Total force magnitude for convergence
    total_force = np.sum(np.abs(forces))
    
    return new_cents, total_force

def _fdl_optimize(cents_init, r_target, max_iters=2000, dt_start=0.005, dt_end=0.0005):
    """
    Run FDL for a fixed r_target.
    """
    cents = cents_init.copy()
    # Adaptive dt
    dt = dt_start
    
    for i in range(max_iters):
        cents, force_mag = _fdl_step(cents, r_target, dt=dt)
        
        # Decay dt
        dt *= 0.995
        
        # Convergence check: if force is very small, break
        if force_mag < 1e-5:
            break
            
    return cents

def _violation_loss(x, r_target):
    """
    Loss function for L-BFGS-B polish.
    Minimizes the sum of squared violations of the constraint d_ij >= 2*r_target and wall >= r_target.
    """
    c = x.reshape(N, 2)
    
    # Wall violations
    wall_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    wall_viol = np.maximum(0, r_target - wall_dists)
    loss = np.sum(wall_viol ** 2)
    
    # Pairwise violations
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = np.maximum(0, 2.0 * r_target - dists)
    loss += np.sum(pair_viol ** 2)
    
    return loss

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Initial Hexagonal Layout
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # 2. FDL Binary Search on Radius
    # We know r0 is feasible. We want to go higher.
    # Upper bound: theoretical max for 50 circles is ~0.0714.
    # Let's set high bound to 0.08 (definitely infeasible) and low to r0.
    
    r_low = r0
    r_high = 0.08
    
    # We will try to find the max r such that FDL converges to a non-overlapping state.
    # How to check feasibility?
    # After FDL, check if min_dist >= 2*r_target - epsilon and wall >= r_target - epsilon.
    
    # Number of binary search steps
    # Each FDL run takes some time. Let's estimate.
    # N=50, O(N^2) per step. 2500 ops. 2000 steps = 5M ops. Fast.
    # We can do many iterations.
    
    # Let's do 20 binary search steps.
    # But we need to be careful: FDL might get stuck in local minima for high r.
    # Strategy: 
    # - Start with r0.
    # - Try to increase r slightly.
    # - If FDL converges to a valid state (no overlaps), accept.
    # - If not, try lower.
    
    # To improve robustness, we can run FDL from the previous best configuration.
    
    current_cents = best_cents.copy()
    
    # Binary search parameters
    num_steps = 25
    # We have ~100s budget. 25 steps * (FDL + Polish) should fit.
    
    for step in range(num_steps):
        if time.time() > deadline - 20:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Run FDL for r_mid
        # Use a moderate number of iterations
        # If we are close to the limit, we might need more iterations to converge.
        # Let's use 1000 iterations for FDL.
        
        t_start = time.time()
        fdl_cents = _fdl_optimize(current_cents, r_mid, max_iters=1000, dt_start=0.002, dt_end=0.0001)
        
        # Check validity
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(fdl_cents[i] - fdl_cents[j])
                if d < min_dist:
                    min_dist = d
        
        wall_min = np.min(np.minimum(0.5 - np.abs(fdl_cents[:, 0]), 0.5 - np.abs(fdl_cents[:, 1])))
        
        actual_r = min(min_dist / 2.0, wall_min)
        
        # If actual_r is close to r_mid, it's feasible
        # We use a small tolerance
        if actual_r >= r_mid - 1e-5:
            # Feasible! Update best and lower bound
            r_low = r_mid
            best_r = max(best_r, actual_r)
            best_cents = fdl_cents.copy()
            current_cents = fdl_cents.copy() # Use this as start for next step
        else:
            # Infeasible
            r_high = r_mid
            # Keep current_cents as is? Or revert to best_cents?
            # Reverting to best_cents might be safer to avoid drifting into bad local minima.
            current_cents = best_cents.copy()
            
        # Optional: Polish the current best with L-BFGS-B occasionally
        # This helps escape FDL local minima.
        if step % 5 == 0 and time.time() < deadline - 30:
            # Polish best_cents for the current r_low
            x0 = best_cents.reshape(-1)
            r_target_polish = r_low
            
            def obj(x):
                return _violation_loss(x, r_target_polish)
                
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                res = minimize(
                    obj, x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                polish_cents = res.x.reshape(N, 2)
                polish_r = _score(polish_cents)
                
                if polish_r > best_r:
                    best_r = polish_r
                    best_cents = polish_cents.copy()
                    current_cents = polish_cents.copy()
                    # If we found a better radius, update r_low
                    if polish_r > r_low:
                        r_low = polish_r
            except Exception:
                pass

    # 3. Final Polish
    if time.time() < deadline - 5:
        x0 = best_cents.reshape(-1)
        r_target_final = best_r
        
        def obj(x):
            return _violation_loss(x, r_target_final)
            
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                obj, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            polish_cents = res.x.reshape(N, 2)
            polish_r = _score(polish_cents)
            
            if polish_r > best_r:
                best_r = polish_r
                best_cents = polish_cents.copy()
        except Exception:
            pass

    # Final safety check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents

# EVOLVE-BLOCK-END
