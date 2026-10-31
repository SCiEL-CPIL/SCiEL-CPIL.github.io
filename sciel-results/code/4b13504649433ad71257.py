"""Combined circle packing: multi-start local optimization + force-directed annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_result = None
    
    def check_validity(x, y, r):
        """Check if packing is valid (all constraints satisfied)."""
        # Boundary check
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
        # Collision check
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - 1e-9:
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
    
    def local_optimize(x0, y0, r0, max_iter=300):
        """Run local optimization from initial guess."""
        z0 = np.concatenate([x0, y0, r0])
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
            
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
    
    def force_anneal(x, y, r, max_steps=2000, initial_temp=0.1, final_temp=0.001):
        """Force-directed simulated annealing to improve packing density."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_sum = np.sum(r)
        
        # Geometric mean radius for scaling forces
        r_mean = np.mean(r)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Cooling schedule (geometric)
            temp = initial_temp * (final_temp / initial_temp) ** (step / max_steps)
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsive forces between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-12:
                        # Strong repulsive force when overlapping
                        overlap = min_dist - dist
                        force = 10.0 * overlap / r_mean
                        fx[i] -= force * dx / dist
                        fy[i] -= force * dy / dist
                        fx[j] += force * dx / dist
                        fy[j] += force * dy / dist
            
            # Boundary forces (keep circles inside [0,1]^2)
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += 5.0 * (r[i] - x[i]) / r_mean
                # Right wall
                if x[i] > 1 - r[i]:
                    fx[i] -= 5.0 * (x[i] - (1 - r[i])) / r_mean
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += 5.0 * (r[i] - y[i]) / r_mean
                # Top wall
                if y[i] > 1 - r[i]:
                    fy[i] -= 5.0 * (y[i] - (1 - r[i])) / r_mean
            
            # Update positions with annealing
            step_size = temp
            x += step_size * fx
            y += step_size * fy
            
            # Clamp to valid range
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            # Try to grow radii slightly if no overlaps
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Grow radii by small amount
                growth = 0.0005 * temp
                new_r = r + growth
                # Check if growth is valid
                valid_growth = True
                for i in range(N):
                    if new_r[i] > min(x[i], 1-x[i], y[i], 1-y[i]):
                        valid_growth = False
                        break
                    for j in range(i + 1, N):
                        d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        if d < new_r[i] + new_r[j] - 1e-9:
                            valid_growth = False
                            break
                    if not valid_growth:
                        break
                
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
        nonlocal best_sum, best_result
        result = local_optimize(x0, y0, r0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))
    
    def polish(x, y, r, max_iter=500):
        """High-precision polish: tighter tolerance optimization."""
        z0 = np.concatenate([x, y, r])
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            r_opt = np.maximum(r_opt, 0.001)
            
            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                # Slight shrink to ensure validity
                r_adjusted = r_opt * 0.999
                if check_validity(x_opt, y_opt, r_adjusted):
                    return x_opt, y_opt, r_adjusted
                else:
                    return x, y, r
        except Exception:
            return x, y, r
    
    # Multi-start optimization with different initializations
    # Hexagonal starts (likely to find good local optima)
    for k in range(20):
        if time.time() > deadline:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.02, 0.02, N)
        y0 += np.random.uniform(-0.02, 0.02, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # Random starts
    for k in range(30):
        if time.time() > deadline:
            break
        x0, y0, r0 = random_init()
        try_optimize(x0, y0, r0)
    
    # Grid starts
    for k in range(5):
        if time.time() > deadline:
            break
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        try_optimize(x0, y0, r0)
    
    # Center-ring starts
    for k in range(10):
        if time.time() > deadline:
            break
        x0, y0, r0 = center_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # Two-ring starts
    for k in range(10):
        if time.time() > deadline:
            break
        x0, y0, r0 = two_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # Force-directed annealing from best result
    if best_result is not None and time.time() < deadline:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]
        
        # Run annealing with different temperature schedules
        for temp_init in [0.05, 0.1, 0.2]:
            if time.time() > deadline:
                break
            x_anneal, y_anneal, r_anneal = force_anneal(x_best, y_best, r_best, 
                                                        max_steps=1500, 
                                                        initial_temp=temp_init)
            current_sum = np.sum(r_anneal)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_anneal, y_anneal, r_anneal))
    
    # Final high-precision polish
    if best_result is not None and time.time() < deadline - 5:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]
        x_polish, y_polish, r_polish = polish(x_best, y_best, r_best, max_iter=300)
        current_sum = np.sum(r_polish)
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = np.column_stack((x_polish, y_polish, r_polish))
    
    # Fallback to seed if no better solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
