import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    # Create grid points
    xs = np.linspace(0.1, 0.9, side)
    ys = np.linspace(0.1, 0.9, side)
    
    # Select N points from grid
    # We want to pick N points that are well distributed.
    # Simple approach: take first N from a flattened grid.
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    grid_xs = grid_xs.flatten()
    grid_ys = grid_ys.flatten()
    
    # Ensure we have at least N points
    if len(grid_xs) < N:
        # Fallback to random if grid is too small (unlikely for N=26)
        grid_xs = np.random.rand(N)
        grid_ys = np.random.rand(N)
    else:
        # Pick N points uniformly from the grid
        indices = np.random.choice(len(grid_xs), N, replace=False)
        grid_xs = grid_xs[indices]
        grid_ys = grid_ys[indices]
        
    # Initial radii: small
    r0 = 0.05
    r = np.full(N, r0)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.concatenate([grid_xs, grid_ys, r])
    
    def objective(v):
        if time.time() > deadline:
            return 1e10 # Stop optimization if time exceeded
            
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Sum of radii (to be maximized, so minimize negative)
        sum_r = np.sum(r)
        
        # Penalties
        penalty = 0.0
        
        # 1. Boundary constraints: x in [r, 1-r], y in [r, 1-r]
        # If x < r, push it up. If x > 1-r, push it down.
        # Penalty: (max(0, r - x) + max(0, x + r - 1))^2
        for i in range(N):
            # x boundary
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > 1.0 - r[i]:
                penalty += (x[i] + r[i] - 1.0)**2
            # y boundary
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > 1.0 - r[i]:
                penalty += (y[i] + r[i] - 1.0)**2
        
        # 2. Collision constraints: distance >= r_i + r_j
        # Penalty: (max(0, r_i + r_j - dist))^2
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # We want to maximize sum_r, so minimize -sum_r + penalty
        # We can scale penalty to ensure it dominates initially
        return -sum_r + 100.0 * penalty

    def gradient(v):
        # Numerical gradient is slow but safe for small N.
        # Let's try analytical or use scipy's default.
        # For simplicity and speed, we'll let scipy handle it or use a simple finite difference.
        # Given N=26, 78 variables, numerical gradient is ~78 evaluations per step.
        # That's too slow for many steps.
        # Let's use a simpler iterative relaxation instead of scipy.optimize.
        # Relaxation is often faster and more robust for packing.
        return None # Not using gradient directly in this approach

    # Let's switch to a physics-based relaxation loop
    # This is more likely to finish in time and converge well.
    
    x = grid_xs.copy()
    y = grid_ys.copy()
    r = np.full(N, 0.05)
    
    dt = 0.01
    max_iter = 5000
    
    best_sum_r = np.sum(r)
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall repulsion
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
            if x[i] > 1.0 - r[i]:
                fx[i] -= (x[i] + r[i] - 1.0) * 10.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
            if y[i] > 1.0 - r[i]:
                fy[i] -= (y[i] + r[i] - 1.0) * 10.0
        
        # Circle repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-9:
                    # Random jitter
                    dx = np.random.rand() - 0.5
                    dy = np.random.rand() - 0.5
                    dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    # Overlap, push apart
                    overlap = min_dist - dist
                    force = overlap * 10.0
                    fx[i] += (dx / dist) * force
                    fy[i] += (dy / dist) * force
                    fx[j] -= (dx / dist) * force
                    fy[j] -= (dy / dist) * force
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Clamp positions to [0, 1] to avoid numerical drift outside
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        # Check for validity and try to increase radii
        # If no overlaps and no boundary violations, we can grow radii
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] > 1.0 - r[i] + 1e-6:
                valid = False
                break
            if y[i] < r[i] - 1e-6 or y[i] > 1.0 - r[i] + 1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Increase radii slightly
            r += 0.001
            # Track best
            current_sum = np.sum(r)
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        else:
            # If invalid, don't grow, just relax positions
            pass
            
    # Final check and return
    # Ensure the returned configuration is valid
    # If not, shrink radii slightly
    for i in range(N):
        best_x[i] = np.clip(best_x[i], best_r[i], 1.0 - best_r[i])
        best_y[i] = np.clip(best_y[i], best_r[i], 1.0 - best_r[i])
        
    # Re-check validity of best
    valid_best = True
    for i in range(N):
        if best_x[i] < best_r[i] or best_x[i] > 1.0 - best_r[i] or best_y[i] < best_r[i] or best_y[i] > 1.0 - best_r[i]:
            valid_best = False
            break
    if valid_best:
        for i in range(N):
            for j in range(i+1, N):
                dx = best_x[i] - best_x[j]
                dy = best_y[i] - best_y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_r[i] + best_r[j]:
                    valid_best = False
                    break
            if not valid_best:
                break
                
    if not valid_best:
        # Fallback to grid if relaxation failed or produced invalid result
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    result = np.column_stack((best_x, best_y, best_r))
    return result

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
