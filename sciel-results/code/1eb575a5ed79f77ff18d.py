import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    num_pairs = len(pairs)
    
    def get_forces_and_energy(x, y, r):
        """
        Compute repulsive forces and potential energy for overlapping circles.
        Uses a simple spring-like model for overlaps: F = k * (penetration).
        Boundary forces push circles back into [0,1]^2.
        """
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-9)
        
        # Overlap: r_i + r_j - dist
        overlap = r[i_idx] + r[j_idx] - dist
        
        # Only apply force if overlapping
        mask = overlap > 0
        if not np.any(mask):
            return np.zeros((N, 2)), 0.0
        
        # Force magnitude proportional to overlap
        # Stiffness k
        k = 50.0
        f_mag = k * overlap[mask]
        
        # Direction: from j to i for i, from i to j for j
        # Normalized direction
        norm_dx = dx[mask] / dist[mask]
        norm_dy = dy[mask] / dist[mask]
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Accumulate forces
        # For each pair (i, j) where i < j:
        # Force on i is in direction (x_i - x_j) / dist
        # Force on j is in direction (x_j - x_i) / dist
        
        ii = i_idx[mask]
        jj = j_idx[mask]
        
        fx[ii] += f_mag * norm_dx
        fy[ii] += f_mag * norm_dy
        
        fx[jj] -= f_mag * norm_dx
        fy[jj] -= f_mag * norm_dy
        
        # Boundary forces
        # If x < r, push right. If x > 1-r, push left.
        # Use a stiff spring for boundaries too to keep them inside.
        kb = 100.0
        
        # Left boundary: x < r
        v_left = r - x
        mask_left = v_left > 0
        if np.any(mask_left):
            fx[mask_left] += kb * v_left[mask_left]
            
        # Right boundary: x > 1 - r
        v_right = x - (1.0 - r)
        mask_right = v_right > 0
        if np.any(mask_right):
            fx[mask_right] -= kb * v_right[mask_right]
            
        # Bottom boundary: y < r
        v_bottom = r - y
        mask_bottom = v_bottom > 0
        if np.any(mask_bottom):
            fy[mask_bottom] += kb * v_bottom[mask_bottom]
            
        # Top boundary: y > 1 - r
        v_top = y - (1.0 - r)
        mask_top = v_top > 0
        if np.any(mask_top):
            fy[mask_top] -= kb * v_top[mask_top]
            
        # Energy calculation (for monitoring, not strictly needed for velocity update if using forces)
        energy = 0.5 * k * np.sum(overlap[mask]**2)
        energy += 0.5 * kb * (np.sum(v_left[mask_left]**2) + np.sum(v_right[mask_right]**2) +
                              np.sum(v_bottom[mask_bottom]**2) + np.sum(v_top[mask_top]**2))
        
        return np.column_stack([fx, fy]), energy

    def simulate(x, y, r, time_limit):
        """
        Run molecular dynamics simulation to relax overlaps.
        """
        m = time.time()
        dt = 0.001
        damping = 0.99
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        # Initial kick to break symmetry if stuck
        vx += np.random.normal(0, 0.01, N)
        vy += np.random.normal(0, 0.01, N)
        
        while time.time() < time_limit:
            forces, energy = get_forces_and_energy(x, y, r)
            
            # Update velocities
            vx = vx * damping + forces[:, 0] * dt
            vy = vy * damping + forces[:, 1] * dt
            
            # Update positions
            x += vx * dt
            y += vy * dt
            
            # Hard clip to prevent explosion (soft boundaries might fail for large dt)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Check for convergence (energy low and forces small)
            if energy < 1e-6 and np.max(np.linalg.norm(forces, axis=1)) < 1e-4:
                break
                
        return x, y

    def project_valid(x, y, r):
        """
        Ensure strict validity by shrinking radii if necessary.
        Returns (x, y, r, is_valid)
        """
        # Boundary check
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_bound)
        
        # Pairwise check
        for _ in range(50):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            if np.max(violations) < 1e-9:
                break
            # Shrink radii to resolve overlap
            # We can shrink more intelligently, but global shrink is safe
            r *= 0.95
            r = np.minimum(r, max_r_bound)
            
        return x, y, r

    def local_refine(x, y, r, time_limit):
        """
        Use Nelder-Mead or L-BFGS-B on a subset of variables or all variables
        to maximize sum(r) subject to constraints.
        Here we use a penalty method with L-BFGS-B for fine tuning.
        """
        z0 = np.concatenate([x, y, r])
        
        def obj(z):
            xi = z[:N]
            yi = z[N:2*N]
            ri = z[2*N:]
            
            # Boundary penalties
            v_l = np.maximum(ri - xi, 0)
            v_r = np.maximum(ri + xi - 1, 0)
            v_b = np.maximum(ri - yi, 0)
            v_t = np.maximum(ri + yi - 1, 0)
            pen_b = np.sum(v_l**2) + np.sum(v_r**2) + np.sum(v_b**2) + np.sum(v_t**2)
            
            # Pairwise penalties
            dx = xi[i_idx] - xi[j_idx]
            dy = yi[i_idx] - yi[j_idx]
            dist_sq = dx*dx + dy*dy
            rs = ri[i_idx] + ri[j_idx]
            pen_p = np.sum(np.maximum(rs**2 - dist_sq, 0)**2)
            
            # Maximize sum(r) -> Minimize -sum(r)
            # Penalty weight must be high enough to enforce constraints
            return -np.sum(ri) + 100.0 * (pen_b + pen_p)
            
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
        
        try:
            res = minimize(
                obj, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success or res.fun < obj(z0):
                x_new = res.x[:N]
                y_new = res.x[N:2*N]
                r_new = res.x[2*N:]
                # Project to ensure validity
                x_new, y_new, r_new = project_valid(x_new, y_new, r_new)
                if np.sum(r_new) > np.sum(r):
                    return x_new, y_new, r_new
        except Exception:
            pass
            
        return x, y, r

    best_x = None
    best_y = None
    best_r = None
    best_score = 0.0
    
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 10.0) # At least 10s per start
    
    for k in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        start_k = time.time()
        sim_limit = start_k + time_per_start * 0.6
        refine_limit = deadline - 5.0
        
        # 1. Initialize
        if k % 2 == 0:
            # Hexagonal-like grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x[count] = (i + 0.5) / side
                        y[count] = (j + 0.5) / side
                        r[count] = 0.08
                        count += 1
        else:
            # Random
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.05)
            
        # Add noise
        x += np.random.normal(0, 0.01, N)
        y += np.random.normal(0, 0.01, N)
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        # 2. Physics Simulation (Relaxation)
        x_sim, y_sim = simulate(x, y, r, sim_limit)
        
        # 3. Project to Validity
        x_valid, y_valid, r_valid = project_valid(x_sim, y_sim, r)
        
        # 4. Local Refinement (Maximize Radii)
        x_ref, y_ref, r_ref = local_refine(x_valid, y_valid, r_valid, refine_limit)
        
        score = np.sum(r_ref)
        if score > best_score:
            best_score = score
            best_x = x_ref
            best_y = y_ref
            best_r = r_ref
            
    # Fallback
    if best_x is None:
        best_x = np.full(N, 0.5)
        best_y = np.full(N, 0.5)
        best_r = np.full(N, 0.01)
        
    # Final Safety Projection
    best_x, best_y, best_r = project_valid(best_x, best_y, best_r)
    
    final_state = np.column_stack([best_x, best_y, best_r])
    return final_state

# EVOLVE-BLOCK-END
