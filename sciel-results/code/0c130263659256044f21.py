import numpy as np
import time
import random

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()
    
    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-8)
        return r

    def check_validity(x, y, r, tol=1e-9):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - tol):
            return False
        return True

    def score_config(x, y):
        r = compute_radii(x, y)
        if check_validity(x, y, r):
            return np.sum(r)
        return -np.inf

    # --- Phase 1: Greedy Gap-Filling Initialization ---
    # This creates a valid, reasonably dense packing by placing circles one by one
    # in the location that maximizes the minimum distance to existing circles and boundaries.
    
    x_init = np.zeros(N)
    y_init = np.zeros(N)
    
    # Helper to find best position for a new circle given existing ones
    def find_best_position(existing_x, existing_y, existing_r, grid_size=50):
        best_pos = (0.5, 0.5)
        best_score = -1.0
        
        # Sample a grid of candidate points
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                cx = i / grid_size
                cy = j / grid_size
                
                # Distance to boundaries
                d_bound = min(cx, 1-cx, cy, 1-cy)
                
                # Distance to existing circles
                min_d = np.inf
                for k in range(len(existing_x)):
                    dx = cx - existing_x[k]
                    dy = cy - existing_y[k]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Effective distance considering existing radius
                    eff_dist = dist - existing_r[k]
                    if eff_dist < min_d:
                        min_d = eff_dist
                        
                # The radius this new circle could have
                potential_r = min(d_bound, 0.5 * min_d)
                
                # We want to maximize the potential radius, but also consider
                # how much "space" it fills. A simple heuristic is just potential_r.
                # To avoid clustering, we might add a small random noise or prefer
                # positions that are further from the centroid of existing circles?
                # For now, maximize potential_r.
                if potential_r > best_score:
                    best_score = potential_r
                    best_pos = (cx, cy)
                    
        # Refine with local search around the best grid point
        cx, cy = best_pos
        for _ in range(10):
            improved = False
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx <= 1 and 0 <= ny <= 1:
                        d_bound = min(nx, 1-nx, ny, 1-ny)
                        min_d = np.inf
                        for k in range(len(existing_x)):
                            ddx = nx - existing_x[k]
                            ddy = ny - existing_y[k]
                            dist = np.sqrt(ddx*ddx + ddy*ddy)
                            eff_dist = dist - existing_r[k]
                            if eff_dist < min_d:
                                min_d = eff_dist
                        potential_r = min(d_bound, 0.5 * min_d)
                        if potential_r > best_score + 1e-6:
                            best_score = potential_r
                            cx, cy = nx, ny
                            improved = True
            if not improved:
                break
                
        return cx, cy, best_score

    for i in range(N):
        if time.time() - start_time > DEADLINE - 100: # Reserve time for SA
            # Fallback to random if too slow
            x_init[i] = random.uniform(0.1, 0.9)
            y_init[i] = random.uniform(0.1, 0.9)
            continue
            
        cx, cy, r_potential = find_best_position(x_init[:i], y_init[:i], compute_radii(x_init[:i], y_init[:i]) if i>0 else np.array([]))
        x_init[i] = cx
        y_init[i] = cy
        
    # Compute initial radii
    r_init = compute_radii(x_init, y_init)
    best_score = np.sum(r_init)
    best_x = x_init.copy()
    best_y = y_init.copy()
    best_r = r_init.copy()
    
    # --- Phase 2: Simulated Annealing ---
    # We treat the positions as the state. The energy is -sum(radii).
    # Moves: Randomly select a circle and move it to a new random position nearby.
    # Acceptance: Metropolis criterion.
    
    current_x = x_init.copy()
    current_y = y_init.copy()
    current_score = best_score
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.0001
    steps = 50000 # Adjust based on time
    
    # Pre-compute time budget for SA
    sa_time_budget = DEADLINE - (time.time() - start_time) - 5
    
    if sa_time_budget > 10:
        # Estimate steps per second roughly
        # Let's just run a fixed number of steps and check time periodically
        steps = 20000
        
        for step in range(steps):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Temperature cooling (geometric)
            alpha = T_end / T_start
            T = T_start * (alpha ** (step / steps))
            
            # Select a random circle to move
            idx = random.randint(0, N-1)
            
            # Save old position
            old_x = current_x[idx]
            old_y = current_y[idx]
            
            # Generate new position (Gaussian step)
            step_size = 0.05 * (0.5 + T / T_start) # Larger steps at high T
            new_x = old_x + random.gauss(0, step_size)
            new_y = old_y + random.gauss(0, step_size)
            
            # Clamp to bounds
            new_x = np.clip(new_x, 0.01, 0.99)
            new_y = np.clip(new_y, 0.01, 0.99)
            
            # Apply move
            current_x[idx] = new_x
            current_y[idx] = new_y
            
            # Calculate new score
            r_new = compute_radii(current_x, current_y)
            
            # Check validity quickly (approximate or full)
            # Full validity check is O(N^2), which is fine for N=26
            if check_validity(current_x, current_y, r_new):
                new_score = np.sum(r_new)
            else:
                # If invalid, the score is effectively -inf for acceptance purposes
                # But we can also use a penalty-based score for SA to allow "repair"
                # For now, let's stick to valid moves only to keep it simple,
                # or use a large penalty.
                # Using penalty is better for SA to escape local minima.
                # Let's calculate penalty-based score.
                dx = current_x[:, None] - current_x[None, :]
                dy = current_y[:, None] - current_y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r_new[:, None] + r_new[None, :]
                np.fill_diagonal(dist, np.inf)
                np.fill_diagonal(min_dist, 0)
                overlaps = np.maximum(0, min_dist - dist)
                penalty = np.sum(overlaps) * 100
                new_score = np.sum(r_new) - penalty
                
            # Metropolis acceptance
            delta_e = new_score - current_score
            if delta_e > 0 or random.random() < np.exp(delta_e / T):
                current_score = new_score
                # Update best if valid
                if check_validity(current_x, current_y, r_new):
                    if new_score > best_score:
                        best_score = new_score
                        best_x = current_x.copy()
                        best_y = current_y.copy()
                        best_r = r_new.copy()
            else:
                # Revert
                current_x[idx] = old_x
                current_y[idx] = old_y
                # current_score remains unchanged
                # r_new is not used, so no need to revert radii explicitly as they are derived

    # --- Phase 3: Local Polish ---
    # Use a simple gradient-free local search to squeeze out more value.
    # Try moving each circle slightly in random directions to improve score.
    
    if time.time() - start_time < DEADLINE - 5:
        x_polish = best_x.copy()
        y_polish = best_y.copy()
        
        for _ in range(1000):
            if time.time() - start_time > DEADLINE - 2:
                break
            improved = False
            for i in range(N):
                for _ in range(10):
                    old_x, old_y = x_polish[i], y_polish[i]
                    # Small random perturbation
                    x_polish[i] += random.gauss(0, 0.005)
                    y_polish[i] += random.gauss(0, 0.005)
                    x_polish[i] = np.clip(x_polish[i], 0.001, 0.999)
                    y_polish[i] = np.clip(y_polish[i], 0.001, 0.999)
                    
                    r_temp = compute_radii(x_polish, y_polish)
                    if check_validity(x_polish, y_polish, r_temp):
                        score_temp = np.sum(r_temp)
                        if score_temp > best_score:
                            best_score = score_temp
                            best_x = x_polish.copy()
                            best_y = y_polish.copy()
                            best_r = r_temp.copy()
                            improved = True
                    else:
                        # Revert if invalid
                        x_polish[i] = old_x
                        y_polish[i] = old_y
            
            if not improved and random.random() < 0.1:
                break # Early stop if no improvement for a while

    # Final validity check and repair
    if not check_validity(best_x, best_y, best_r):
        # Scale down radii until valid
        for scale in np.linspace(0.999, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)
            if not check_validity(best_x, best_y, best_r):
                # Last resort: return a simple valid grid
                x_grid = np.array([(j + 0.5) / 5.0 for i in range(5) for j in range(5)] + [0.5])
                y_grid = np.array([(i + 0.5) / 5.0 for i in range(5) for j in range(5)] + [0.5])
                best_x = x_grid
                best_y = y_grid
                best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
