import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Physics-based simulation parameters
    # We model circles as particles.
    # Forces:
    # 1. Repulsion if overlapping: F_rep = k_rep * (r_i + r_j - dist) * direction
    # 2. Attraction to center: F_att = k_att * (center - pos)
    # 3. Wall repulsion: F_wall = k_wall * (r_i - dist_to_wall) * normal
    
    # We will simulate "growth" of radii or just optimize positions with fixed radii?
    # The task asks to maximize sum of radii.
    # A common trick: Fix radii to a target value R, and try to pack them.
    # If we can pack N circles of radius R, sum = N*R.
    # We can binary search on R.
    # Or, we can let radii be variables in the simulation?
    # Let's try the "Fixed Radius Binary Search" approach as it simplifies the dynamics.
    # We just need to find the max R such that N circles of radius R fit.
    
    # Estimate max R:
    # Hexagonal packing density ~0.9069.
    # Area of square = 1. Area of circles = N * pi * R^2.
    # N * pi * R^2 <= 0.9 * 1  => R <= sqrt(0.9 / (N * pi))
    # For N=26: R <= sqrt(0.9 / (26*3.14)) ~ sqrt(0.011) ~ 0.105
    # Let's search R in [0.05, 0.11]
    
    def simulate_packing(radius, seed=0):
        """
        Attempt to pack N circles of given 'radius' inside [0,1]^2.
        Returns True if a valid packing is found, False otherwise.
        """
        rng = np.random.default_rng(seed)
        
        # Initialize positions randomly
        # Ensure they don't start too close to walls
        margin = radius
        if 2*margin >= 1.0:
            return False
            
        x = rng.uniform(margin, 1.0 - margin, N)
        y = rng.uniform(margin, 1.0 - margin, N)
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        # Simulation parameters
        dt = 0.005
        k_rep = 10.0  # Repulsion strength
        k_att = 0.1   # Attraction to center strength (keeps them compact)
        k_wall = 20.0 # Wall repulsion strength
        damping = 0.9 # Velocity damping (friction)
        
        # Cooling schedule: reduce damping over time? Or just run for fixed steps.
        # Let's run for a fixed number of steps and check convergence.
        max_steps = 2000
        
        for step in range(max_steps):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # 1. Pairwise Repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    min_dist = 2.0 * radius
                    
                    if dist < min_dist and dist > 1e-9:
                        # Overlap
                        overlap = min_dist - dist
                        # Force direction: from j to i
                        f_mag = k_rep * overlap
                        fx[i] += f_mag * (dx / dist)
                        fy[i] += f_mag * (dy / dist)
                        fx[j] -= f_mag * (dx / dist)
                        fy[j] -= f_mag * (dy / dist)
                    elif dist == 0:
                        # Coincident, push apart randomly
                        fx[i] += rng.uniform(-1, 1)
                        fy[i] += rng.uniform(-1, 1)
                        fx[j] -= rng.uniform(-1, 1)
                        fy[j] -= rng.uniform(-1, 1)
            
            # 2. Wall Repulsion
            # x < r
            for i in range(N):
                if x[i] < radius:
                    fx[i] += k_wall * (radius - x[i])
                elif x[i] > 1.0 - radius:
                    fx[i] -= k_wall * (x[i] - (1.0 - radius))
                
                if y[i] < radius:
                    fy[i] += k_wall * (radius - y[i])
                elif y[i] > 1.0 - radius:
                    fy[i] -= k_wall * (y[i] - (1.0 - radius))
            
            # 3. Attraction to Center (0.5, 0.5)
            # This helps prevent circles from drifting to corners and getting stuck in bad local minima
            # It encourages a compact cluster.
            fx += k_att * (0.5 - x)
            fy += k_att * (0.5 - y)
            
            # Update velocities and positions
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            
            x += vx * dt
            y += vy * dt
            
            # Hard clamp to prevent escaping the square entirely (safety)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Check for convergence/stability
            # If max velocity is very low and no overlaps, we can stop early?
            # For simplicity, run all steps.
            
        # Final Validation
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 2.0 * radius - 1e-6:
                    return False, None
        
        # Check walls
        for i in range(N):
            if x[i] < radius - 1e-6 or x[i] > 1.0 - radius + 1e-6:
                return False, None
            if y[i] < radius - 1e-6 or y[i] > 1.0 - radius + 1e-6:
                return False, None
                
        return True, np.column_stack((x, y, np.full(N, radius)))

    # Binary Search for Max Radius
    # Lower bound: something feasible. 0.05 is safe for 26 circles?
    # 26 * pi * 0.05^2 ~ 26 * 3.14 * 0.0025 ~ 0.2. Density 0.2. Very safe.
    r_low = 0.05
    r_high = 0.115 # Slightly above theoretical hex limit
    
    best_result = None
    best_val = 0.0
    
    # We will do a coarse binary search first, then refine.
    # To save time, we can use a few random restarts for each radius test.
    
    def try_radius(r, num_restarts=3):
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            success, result = simulate_packing(r, seed=seed)
            if success:
                val = np.sum(result[:, 2])
                if best_result is None or val > best_val:
                    # Update global best
                    # Note: simulate_packing returns the result, we need to update outer scope
                    # Since Python closures are tricky with assignment, let's return the result
                    return result
        return None

    # Binary Search Steps
    # We want to find the largest r such that try_radius(r) is not None.
    # We'll do 10-15 iterations of binary search.
    
    # First, ensure r_low is feasible
    res_low = try_radius(r_low, num_restarts=5)
    if res_low is None:
        # Fallback if even low radius fails (shouldn't happen)
        # Return a grid
        n_x, n_y = 6, 5
        r_val = 0.07
        x_coords = np.linspace(r_val, 1-r_val, n_x)
        y_coords = np.linspace(r_val, 1-r_val, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()
        y_f = y_f.flatten()
        r_f = np.full(n_x*n_y, r_val)
        return np.column_stack((x_f[:N], y_f[:N], r_f[:N]))
    
    best_result = res_low
    best_val = np.sum(res_low[:, 2])
    
    for _ in range(12): # ~4096 search space resolution
        if time.time() > deadline - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to pack with r_mid
        # Use more restarts for the mid-point to be more confident
        res_mid = try_radius(r_mid, num_restarts=5)
        
        if res_mid is not None:
            # Feasible
            r_low = r_mid
            val_mid = np.sum(res_mid[:, 2])
            if val_mid > best_val:
                best_val = val_mid
                best_result = res_mid
        else:
            # Infeasible
            r_high = r_mid
            
    # Final Polish:
    # We have a feasible radius r_low.
    # Let's try to optimize the specific configuration found at r_low slightly more
    # or just return it. The binary search converges quickly.
    # To be safe, let's do one more check at r_low with more restarts to see if we can get a better config
    # (though for fixed radius, the value is the same N*r_low, so position doesn't change the score, only validity).
    # Wait, the score is sum of radii. If we fix radius, score is fixed.
    # The binary search finds the max R.
    
    # One potential issue: The simulation might fail for a radius that is actually feasible due to local minima.
    # The binary search might underestimate R if it fails at R* but succeeds at R*-epsilon.
    # To mitigate, we used multiple restarts.
    
    if best_result is not None:
        return best_result
    else:
        # Absolute fallback
        n_x, n_y = 6, 5
        r_val = 0.07
        x_coords = np.linspace(r_val, 1-r_val, n_x)
        y_coords = np.linspace(r_val, 1-r_val, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()
        y_f = y_f.flatten()
        r_f = np.full(n_x*n_y, r_val)
        return np.column_stack((x_f[:N], y_f[:N], r_f[:N]))

# EVOLVE-BLOCK-END
