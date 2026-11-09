"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.spatial.distance import cdist
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    # Compute the score (radius) for a given set of centers
    def compute_radius(centers):
        """Compute the achievable radius for given centers."""
        # Min pairwise distance / 2
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Min distance to border: min over i of min(0.5 - |x_i|, 0.5 - |y_i|)
        dist_border = np.min(0.5 - np.abs(centers))
        return min(min_pair, dist_border)

    # Generate initial configuration: use a hexagonal-like pattern
    def make_initial_config():
        """Create a good initial configuration using hexagonal packing."""
        # Hexagonal lattice generation
        # For 50 circles, try different row arrangements
        # Best known packing has r ≈ 0.071377, so spacing ≈ 0.14275
        
        centers = []
        spacing = 0.143
        row_height = spacing * np.sqrt(3) / 2
        
        # Try to fit rows
        y = -0.45
        row_idx = 0
        while y < 0.5 and len(centers) < N:
            x_offset = (spacing * 0.5) if (row_idx % 2 == 1) else 0.0
            x = -0.45 + x_offset
            while x < 0.5 and len(centers) < N:
                centers.append([x, y])
                x += spacing
            y += row_height
            row_idx += 1
        
        # If we have more than N, take the first N
        # If we have less, pad with random points
        centers = np.array(centers[:N])
        
        if len(centers) < N:
            # Add random points to fill
            extra = np.random.uniform(-0.4, 0.4, size=(N - len(centers), 2))
            centers = np.vstack([centers, extra])
        
        # Add small random perturbation to break symmetry
        centers += np.random.normal(0, 0.001, size=centers.shape)
        
        # Clip to stay in bounds
        centers = np.clip(centers, -0.49, 0.49)
        
        return centers

    # Simulated annealing optimization
    def simulated_annealing(centers, max_time=140.0):
        """Optimize using simulated annealing with local moves."""
        current = centers.copy()
        current_r = compute_radius(current)
        best = current.copy()
        best_r = current_r
        
        # Temperature schedule
        T = 0.01
        T_min = 0.00001
        cooling = 0.995
        
        t0 = time.time()
        rng = np.random.RandomState(42)
        
        while time_ok() and T > T_min:
            # Generate a random move
            n_move = rng.randint(1, 4)  # Move 1-3 points
            indices = rng.choice(N, size=n_move, replace=False)
            
            new = current.copy()
            for idx in indices:
                # Random displacement
                dx = rng.uniform(-0.02, 0.02)
                dy = rng.uniform(-0.02, 0.02)
                new[idx] += [dx, dy]
            
            # Keep in bounds
            new = np.clip(new, -0.49, 0.49)
            
            new_r = compute_radius(new)
            delta = new_r - current_r
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or rng.random() < np.exp(delta / T):
                current = new
                current_r = new_r
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
            
            T *= cooling
            
            # Periodic restart if stuck
            if rng.random() < 0.001:
                T = max(T * 2, 0.005)
        
        return best, best_r

    # Local optimization: push apart closest pairs and inward from borders
    def local_optimize(centers, max_iters=5000):
        """Local optimization by pushing constraining points apart."""
        current = centers.copy()
        current_r = compute_radius(current)
        best = current.copy()
        best_r = current_r
        
        t0 = time.time()
        for it in range(max_iters):
            if not time_ok():
                break
            
            D = cdist(current, current)
            np.fill_diagonal(D, np.inf)
            min_pair = np.min(D) / 2.0
            dist_border = np.min(0.5 - np.abs(current))
            r = min(min_pair, dist_border)
            
            if r > best_r:
                best = current.copy()
                best_r = r
            
            # Find constraining constraints
            # Closest pair
            idx_min = np.unravel_index(np.argmin(D), D.shape)
            i, j = idx_min[0], idx_min[1]
            
            # Push apart
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                push = diff / dist * 0.001
                current[i] += push
                current[j] -= push
            
            # Points too close to border
            for k in range(N):
                for coord in range(2):
                    if current[k, coord] > 0.5 - r - 0.001:
                        current[k, coord] = 0.5 - r - 0.002
                    elif current[k, coord] < -0.5 + r + 0.001:
                        current[k, coord] = -0.5 + r + 0.002
            
            # Keep in bounds
            current = np.clip(current, -0.49, 0.49)
            
            # If no improvement in a while, add small random perturbation
            if it % 100 == 0 and it > 0:
                current += np.random.normal(0, 0.002, size=current.shape)
                current = np.clip(current, -0.49, 0.49)
        
        return best, best_r

    # Bisection on radius with penalty-based feasibility checks
    def bisection_radius(centers, max_iters=100):
        """Use bisection to find the maximum feasible radius."""
        current = centers.copy()
        
        # Initial bounds
        r_low = 0.05
        r_high = 0.08
        
        # Check if r_high is feasible
        def is_feasible(r):
            """Check if circles of radius r can fit at the given centers."""
            # Check pairwise distances
            D = cdist(current, current)
            np.fill_diagonal(D, np.inf)
            if np.min(D) < 2 * r - 1e-10:
                return False
            # Check boundary distances
            if np.min(0.5 - np.abs(current)) < r - 1e-10:
                return False
            return True
        
        # If r_high is not feasible, decrease it
        while not is_feasible(r_high) and r_high > r_low:
            r_high = (r_high + r_low) / 2
        
        # Now bisect
        for _ in range(max_iters):
            if not time_ok():
                break
            
            r_mid = (r_low + r_high) / 2
            
            if is_feasible(r_mid):
                r_low = r_mid
            else:
                r_high = r_mid
            
            if r_high - r_low < 1e-8:
                break
        
        return r_low

    # Main optimization loop with multiple strategies
    best_centers = None
    best_r = -1.0
    
    # Try multiple initial configurations
    n_attempts = 5
    for attempt in range(n_attempts):
        if not time_ok():
            break
        
        if attempt == 0:
            centers = make_initial_config()
        else:
            # Perturb the best found so far
            if best_centers is not None:
                centers = best_centers + np.random.normal(0, 0.01, size=(N, 2))
            else:
                centers = make_initial_config()
            centers = np.clip(centers, -0.49, 0.49)
        
        # Run simulated annealing
        sa_centers, sa_r = simulated_annealing(centers)
        
        # Run local optimization on the SA result
        lo_centers, lo_r = local_optimize(sa_centers)
        
        # Also try local optimization on the original
        lo2_centers, lo2_r = local_optimize(centers)
        
        # Use bisection to refine the radius for the best local optimization result
        if lo_r > best_r:
            refined_r = bisection_radius(lo_centers)
            if refined_r > lo_r:
                lo_r = refined_r
        
        if lo2_r > best_r:
            refined_r2 = bisection_radius(lo2_centers)
            if refined_r2 > lo2_r:
                lo2_r = refined_r2
        
        # Keep the best
        for c, r in [(sa_centers, sa_r), (lo_centers, lo_r), (lo2_centers, lo2_r)]:
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        
        if not time_ok():
            break
    
    # Final validation
    if best_centers is None:
        best_centers = make_initial_config()
    
    result = best_centers.copy()
    
    # Ensure all values are finite and within bounds
    if not np.all(np.isfinite(result)):
        result = make_initial_config()
    
    result = np.clip(result, -0.49, 0.49)
    
    return result

# EVOLVE-BLOCK-END
