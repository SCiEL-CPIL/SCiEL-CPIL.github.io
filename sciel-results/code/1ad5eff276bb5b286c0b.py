"""Combined circle packing: multi-start local optimization + force-directed annealing + high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (5s buffer)
    
    best_sum = 0.0
    best_result = None
    best_x = None
    best_y = None
    best_r = None

    def check_validity(x, y, r, tol=1e-9):
        """Check if packing is valid (all constraints satisfied)."""
        # Boundary check
        if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Collision check
        # Vectorized distance check for speed
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r[:, None] + r[None, :]
        # Check if distance is less than sum of radii (with tolerance)
        # We check if dist_sq < (r_i + r_j)^2
        # To avoid sqrt, we compare squares. But careful with tol.
        # Actually, simple loop is safer for strict validity if N is small, but vectorized is faster.
        # Let's stick to the robust loop for the final check to be 100% sure, 
        # but use vectorized for the objective function.
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def objective(z, penalty_weight=1000.0):
        """Objective: minimize -sum(r) + penalty for violations."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        
        # Clip radii to positive values
        r_safe = np.maximum(r, 0.001)
        
        sum_r = np.sum(r_safe)
        
        # Boundary violations
        v_bound = 0.0
        v_bound += np.sum(np.maximum(0, r_safe - x)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - x))**2)
        v_bound += np.sum(np.maximum(0, r_safe - y)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - y))**2)
        
        # Collision violations - vectorized for speed
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r_safe[:, None] + r_safe[None, :]
        overlap_terms = np.maximum(0, sum_r_mat**2 - dist_sq)
        np.fill_diagonal(overlap_terms, 0)
        v_coll = np.sum(overlap_terms**2)
        
        return -sum_r + penalty_weight * (v_bound + v_coll)

    def local_optimize(x0, y0, r0, max_iter=300, penalty_weight=1000.0, ftol=1e-8, gtol=1e-6):
        """Run local optimization from initial guess."""
        z0 = np.concatenate([x0, y0, r0])
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                          args=(penalty_weight,),
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            # Post-process: shrink radii slightly to ensure validity
            r_opt = np.maximum(r_opt, 0.001)
            
            # Check validity and adjust if needed
            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                # Try to fix by shrinking radii
                scale = 0.95
                r_adjusted = r_opt * scale
                if check_validity(x_opt, y_opt, r_adjusted):
                    return x_opt, y_opt, r_adjusted
                else:
                    # More aggressive repair
                    scale = 0.90
                    r_adjusted = r_opt * scale
                    if check_validity(x_opt, y_opt, r_adjusted):
                        return x_opt, y_opt, r_adjusted
                    else:
                        return None
        except Exception:
            return None

    def force_anneal(x, y, r, time_budget=2.0):
        """
        Force-directed simulated annealing.
        Uses vectorized force calculations for speed.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_sum = np.sum(r)
        
        r_mean = np.mean(r)
        
        # Parameters
        initial_temp = 0.05
        final_temp = 0.001
        max_steps = 1000 # Will break based on time
        force_scale = 5.0
        
        start_fa = time.time()
        
        for step in range(max_steps):
            if time.time() > start_fa + time_budget:
                break
            
            # Cooling schedule
            temp = initial_temp * (final_temp / initial_temp) ** (step / max_steps)
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsive forces
            # Vectorized calculation of overlaps
            x_diff = x[:, None] - x[None, :]
            y_diff = y[:, None] - y[None, :]
            dist_sq = x_diff**2 + y_diff**2
            # Avoid division by zero
            dist_sq_safe = np.where(dist_sq < 1e-12, 1e-12, dist_sq)
            dist = np.sqrt(dist_sq_safe)
            
            sum_r_mat = r[:, None] + r[None, :]
            overlap = np.maximum(0, sum_r_mat - dist)
            
            # Force magnitude: proportional to overlap
            # F = k * overlap
            # Direction: along the line connecting centers
            # We need to sum forces on each particle
            # force_vec_ij = force_mag * (r_j - r_i) / dist
            # But we only want repulsive if overlapping.
            
            # Matrix of force magnitudes
            # Note: overlap is symmetric.
            # We compute the force component on i due to j.
            # F_ij_x = -force_scale * overlap_ij * (x_j - x_i) / dist_ij
            # Since overlap is 0 if not overlapping, this works.
            
            # Be careful with diagonal (dist=0). We set dist_sq_safe to 1e-12, so dist is small.
            # But overlap on diagonal is 0 (sum_r - 0? No, sum_r is r_i+r_i, dist is 0. 
            # Wait, dist_sq_safe is 1e-12, so dist is 1e-6. 
            # sum_r is 2*r. overlap = 2*r - 1e-6 > 0.
            # This creates a huge force on the diagonal!
            # We must zero out the diagonal of the overlap matrix.
            np.fill_diagonal(overlap, 0)
            
            # Force vector components
            # dx = x_j - x_i. In matrix form: x[None, :] - x[:, None] gives x_j - x_i?
            # x[:, None] is column i. x[None, :] is row j.
            # x[None, :] - x[:, None] is x_j - x_i. Correct.
            dx_mat = x[None, :] - x[:, None]
            dy_mat = y[None, :] - y[:, None]
            
            # Force magnitude matrix
            f_mag = force_scale * overlap / r_mean
            
            # Force vector matrix
            # F_x_ij = f_mag * dx / dist
            f_x_mat = f_mag * dx_mat / dist
            f_y_mat = f_mag * dy_mat / dist
            
            # Sum forces on each particle i
            # Force on i is sum over j of F_ij
            # Note: F_ij is force on i due to j.
            # The matrix f_x_mat[i, j] is the x-component of force on i due to j.
            # So fx = np.sum(f_x_mat, axis=1)
            
            fx = np.sum(f_x_mat, axis=1)
            fy = np.sum(f_y_mat, axis=1)
            
            # Boundary forces
            # Left: if x < r, push right (+)
            # Right: if x > 1-r, push left (-)
            # Bottom: if y < r, push up (+)
            # Top: if y > 1-r, push down (-)
            
            # Vectorized boundary forces
            # Left wall
            left_viol = np.maximum(0, r - x)
            fx += 5.0 * left_viol / r_mean
            
            # Right wall
            right_viol = np.maximum(0, x - (1 - r))
            fx -= 5.0 * right_viol / r_mean
            
            # Bottom wall
            bottom_viol = np.maximum(0, r - y)
            fy += 5.0 * bottom_viol / r_mean
            
            # Top wall
            top_viol = np.maximum(0, y - (1 - r))
            fy -= 5.0 * top_viol / r_mean
            
            # Update positions
            step_size = temp
            x += step_size * fx
            y += step_size * fy
            
            # Clamp to valid range (hard constraint)
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            # Try to grow radii if valid
            # We do this less frequently to save time, or just check validity
            if step % 10 == 0:
                if check_validity(x, y, r, tol=1e-7):
                    # Grow radii slightly
                    growth = 0.0002 * temp
                    new_r = r + growth
                    # Check if growth is valid
                    # Simple check: if new_r is within bounds and no collisions
                    # We'll do a quick check
                    valid_growth = True
                    # Boundary check
                    if np.any(new_r > np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))):
                        valid_growth = False
                    else:
                        # Collision check (simplified)
                        x_diff = x[:, None] - x[None, :]
                        y_diff = y[:, None] - y[None, :]
                        dist_sq = x_diff**2 + y_diff**2
                        sum_r_mat = new_r[:, None] + new_r[None, :]
                        # Check if dist_sq < sum_r^2
                        # We can vectorize this check
                        overlap_check = np.maximum(0, sum_r_mat**2 - dist_sq)
                        np.fill_diagonal(overlap_check, 0)
                        if np.any(overlap_check > 1e-8):
                            valid_growth = False
                    
                    if valid_growth:
                        r = new_r
            
            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        return best_x, best_y, best_r

    def hexagonal_init(n_circles):
        """Create hexagonal packing initialization."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        # Hexagonal arrangement
        row_spacing = 0.866  # sqrt(3)/2
        col_spacing = 1.0
        
        # Estimate grid size
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        
        # Scale to fit in unit square
        total_width = (cols - 1) * col_spacing
        total_height = (rows - 1) * row_spacing
        
        # Add margins
        margin_x = 0.05
        margin_y = 0.05
        
        scale_x = (1 - 2*margin_x) / max(total_width, 1e-6)
        scale_y = (1 - 2*margin_y) / max(total_height, 1e-6)
        scale = min(scale_x, scale_y, 1.0)
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x[i] = margin_x + col * col_spacing * scale + (row % 2) * 0.5 * col_spacing * scale
            y[i] = margin_y + row * row_spacing * scale
            r[i] = 0.5 * min(col_spacing, row_spacing) * scale * 0.95
        
        # Ensure radii don't exceed bounds
        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def random_init():
        """Random initialization."""
        r = np.random.uniform(0.03, 0.1, N)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        return x, y, r

    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.95
        
        return x, y, r

    def center_ring_init():
        """Center-ring initialization: one circle in center, rest in a ring."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Center circle
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.12
        
        # Ring of remaining circles
        n_ring = N - 1
        ring_radius = 0.35
        for i in range(1, N):
            angle = 2 * np.pi * (i - 1) / n_ring
            x[i] = 0.5 + ring_radius * np.cos(angle)
            y[i] = 0.5 + ring_radius * np.sin(angle)
            r[i] = 0.08
        
        # Ensure within bounds
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def two_ring_init():
        """Two-ring initialization: center + inner ring + outer ring."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Center circle
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.10
        
        # Inner ring: 8 circles
        n_inner = 8
        inner_radius = 0.25
        for i in range(1, 1 + n_inner):
            angle = 2 * np.pi * (i - 1) / n_inner
            x[i] = 0.5 + inner_radius * np.cos(angle)
            y[i] = 0.5 + inner_radius * np.sin(angle)
            r[i] = 0.07
        
        # Outer ring: remaining circles
        n_outer = N - 1 - n_inner
        outer_radius = 0.42
        for i in range(1 + n_inner, N):
            idx = i - 1 - n_inner
            angle = 2 * np.pi * idx / n_outer + np.pi / n_outer
            x[i] = 0.5 + outer_radius * np.cos(angle)
            y[i] = 0.5 + outer_radius * np.sin(angle)
            r[i] = 0.06
        
        # Ensure within bounds
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def try_optimize(x0, y0, r0):
        """Try to optimize from given initial state, return result if better."""
        nonlocal best_sum, best_result, best_x, best_y, best_r
        
        # Stage 1: Coarse L-BFGS
        if time.time() > deadline - 15: return
        res1 = local_optimize(x0, y0, r0, max_iter=100, penalty_weight=100.0, ftol=1e-5, gtol=1e-4)
        if res1 is None: return
        x1, y1, r1 = res1
        
        # Stage 2: Medium L-BFGS
        if time.time() > deadline - 15:
            result = res1
        else:
            res2 = local_optimize(x1, y1, r1, max_iter=200, penalty_weight=1000.0, ftol=1e-7, gtol=1e-5)
            if res2 is not None:
                x2, y2, r2 = res2
                if np.sum(r2) >= np.sum(r1):
                    result = res2
                else:
                    result = res1
            else:
                result = res1
        
        # Stage 3: Force Directed Annealing (Escape local optima)
        x_res, y_res, r_res = result
        if time.time() > deadline - 20:
            pass
        else:
            # Allocate 2 seconds for FDA
            fa_budget = min(2.0, deadline - 20 - time.time())
            if fa_budget > 0.5:
                x_fa, y_fa, r_fa = force_anneal(x_res, y_res, r_res, time_budget=fa_budget)
                if np.sum(r_fa) > np.sum(r_res):
                    result = (x_fa, y_fa, r_fa)
        
        # Stage 4: High-Precision Polish
        x_res, y_res, r_res = result
        if time.time() > deadline - 5:
            pass
        else:
            # Very tight tolerances and high penalty
            res4 = local_optimize(x_res, y_res, r_res, max_iter=500, penalty_weight=100000.0, ftol=1e-12, gtol=1e-10)
            if res4 is not None:
                x4, y4, r4 = res4
                if np.sum(r4) >= np.sum(r_res):
                    result = res4
        
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))
                best_x, best_y, best_r = x_opt, y_opt, r_opt

    # Multi-start optimization
    # Reduced number of restarts to allow time for FDA and Polish
    
    # Hexagonal starts
    for k in range(6):
        if time.time() > deadline - 25:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.02, 0.02, N)
        y0 += np.random.uniform(-0.02, 0.02, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # Random starts
    for k in range(8):
        if time.time() > deadline - 25:
            break
        x0, y0, r0 = random_init()
        try_optimize(x0, y0, r0)
    
    # Grid starts
    for k in range(2):
        if time.time() > deadline - 25:
            break
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        try_optimize(x0, y0, r0)
    
    # Center-ring starts
    for k in range(3):
        if time.time() > deadline - 25:
            break
        x0, y0, r0 = center_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # Two-ring starts
    for k in range(3):
        if time.time() > deadline - 25:
            break
        x0, y0, r0 = two_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Final Polish on the best result found so far, if time permits
    if best_result is not None and time.time() < deadline - 2:
        x_best, y_best, r_best = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        # One more polish pass on the absolute best
        res_final = local_optimize(x_best, y_best, r_best, max_iter=1000, penalty_weight=1000000.0, ftol=1e-14, gtol=1e-12)
        if res_final is not None:
            xf, yf, rf = res_final
            if np.sum(rf) > best_sum:
                best_sum = np.sum(rf)
                best_result = np.column_stack((xf, yf, rf))

    # Fallback to seed if no better solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
