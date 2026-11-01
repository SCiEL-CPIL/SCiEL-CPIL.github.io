"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Penalty Method."""
import numpy as np
import time

N = 21
_TIME_LIMIT = 145.0

def _is_valid(x, y, r, w, h, tol=1e-7):
    """Check validity of the configuration."""
    if w <= 0.05 or h <= 0.05:
        return False
    # Check walls
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    # Check pairs
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True

def _value(x, y, r):
    return float(np.sum(r))

def _check_and_get_value(x, y, r, w, h, tol=1e-7):
    if _is_valid(x, y, r, w, h, tol):
        return True, _value(x, y, r)
    return False, 0.0

def _energy(x, y, r, w, h, lambda_wall=100.0, lambda_pair=100.0):
    """
    Compute a smooth energy landscape for optimization.
    We want to minimize E = -sum(r) + penalties.
    """
    energy = -np.sum(r)
    
    # Wall penalties (smooth)
    # x_i < r_i -> penalty (r_i - x_i)^2
    # x_i > w - r_i -> penalty (x_i - (w - r_i))^2
    # Same for y
    
    # Vectorized wall penalties
    # Left/Right
    pen_x_left = np.maximum(0, r - x)**2
    pen_x_right = np.maximum(0, x - (w - r))**2
    pen_y_bottom = np.maximum(0, r - y)**2
    pen_y_top = np.maximum(0, y - (h - r))**2
    
    energy += lambda_wall * (np.sum(pen_x_left) + np.sum(pen_x_right) + np.sum(pen_y_bottom) + np.sum(pen_y_top))
    
    # Pair penalties
    # d < r_i + r_j -> penalty (r_i + r_j - d)^2
    # We compute this in a loop for clarity, or vectorize.
    # Vectorization is faster.
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx*dx + dy*dy
    # Avoid sqrt(0) issues, though d=0 is rare in random starts
    d = np.sqrt(d_sq + 1e-12)
    r_sum = r[:, None] + r[None, :]
    
    # Only consider i < j to avoid double counting and self-interaction
    # Mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    overlap = np.maximum(0, r_sum - d)
    energy += lambda_pair * np.sum(overlap[mask]**2)
    
    return energy

def _construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_state = None
    
    # Initial Guess: Hexagonal-ish packing in a square
    # Square side = 1.0 (perimeter 4)
    w0 = 1.0
    h0 = 1.0
    
    # Create a grid of 5x5 = 25 cells, pick 21
    # Or just random placement
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start Simulated Annealing
    n_starts = 15
    time_per_start = (_TIME_LIMIT - 5) / n_starts
    
    for start_idx in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > _TIME_LIMIT - 10:
            break
            
        # Initialize state
        # Random aspect ratio
        w = rng.uniform(0.8, 1.2)
        h = 2.0 - w
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Place circles randomly, ensuring initial non-overlap if possible
        # Simple random placement with small radii
        for i in range(N):
            r[i] = rng.uniform(0.05, 0.1)
            x[i] = rng.uniform(r[i], w - r[i])
            y[i] = rng.uniform(r[i], h - r[i])
            
        # If overlap, push apart slightly or just let energy handle it
        # Let's do a quick greedy placement for better start
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Greedy placement
        # Sort by some heuristic? Or just random order.
        # Let's try to place them one by one.
        # We need to handle the variable w/h.
        # Let's fix w/h for the SA run? No, let's optimize w too.
        # But w optimization is tricky in SA.
        # Let's fix w=1.0 for the SA, then optimize w in a final pass?
        # Or include w in the state.
        # State: x, y, r, w.
        # Dimensions: 3*N + 1 = 64.
        
        # Let's stick to fixed aspect ratio w=1.0 for the SA to reduce complexity,
        # then do a 1D optimization on w at the end?
        # Actually, the optimal aspect ratio might not be 1.0.
        # Let's include w in the SA state.
        
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.05, 0.1, N)
        w = 1.0
        h = 1.0
        
        # Ensure initial validity (or close to it)
        # If invalid, the energy will be high, which is fine for SA.
        
        current_energy = _energy(x, y, r, w, h)
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.001
        cooling_rate = 0.995
        step_size = 0.05
        max_iter = 20000 # Adjust based on time
        
        # Estimate iterations allowed
        # 20000 iterations * 64 vars * (calc energy) ~ fast enough?
        # _energy is O(N^2) = 441 ops. 20000 * 441 ~ 10M ops. Python loop might be slow.
        # Let's check speed.
        # 10M ops in Python ~ 1-2 seconds.
        # So 15 starts * 2s = 30s. We have 145s.
        # We can afford more iterations or more starts.
        
        # Let's do 50000 iterations per start.
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Perturb one variable
            var_idx = rng.integers(0, 3*N + 1)
            delta = rng.normal(0, step_size)
            
            old_val = 0
            if var_idx < 3*N:
                if var_idx % 3 == 0: # x
                    old_val = x[var_idx//3]
                    x[var_idx//3] += delta
                elif var_idx % 3 == 1: # y
                    old_val = y[var_idx//3]
                    y[var_idx//3] += delta
                else: # r
                    old_val = r[var_idx//3]
                    r[var_idx//3] += delta
                    if r[var_idx//3] < 0.01:
                        r[var_idx//3] = 0.01
                        delta = 0 # revert if hit bound
            else: # w
                old_val = w
                w += delta
                if w < 0.5 or w > 1.5:
                    w = old_val
                    delta = 0
            
            if delta == 0:
                continue
                
            h = 2.0 - w
            if h < 0.1:
                # Revert
                if var_idx < 3*N:
                    if var_idx % 3 == 0: x[var_idx//3] = old_val
                    elif var_idx % 3 == 1: y[var_idx//3] = old_val
                    else: r[var_idx//3] = old_val
                else:
                    w = old_val
                continue
                
            new_energy = _energy(x, y, r, w, h)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
            else:
                # Revert
                if var_idx < 3*N:
                    if var_idx % 3 == 0: x[var_idx//3] = old_val
                    elif var_idx % 3 == 1: y[var_idx//3] = old_val
                    else: r[var_idx//3] = old_val
                else:
                    w = old_val
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Maybe restart or reduce step size
                step_size *= 0.9
            
            # Check validity and update best
            # This is expensive (O(N^2)). Do it every 100 steps.
            if it % 100 == 0:
                valid, val = _check_and_get_value(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_state = (x.copy(), y.copy(), r.copy(), w)
        
        # End of SA for this start
        # Check final state
        valid, val = _check_and_get_value(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_state = (x.copy(), y.copy(), r.copy(), w)
            
    # Final Polish: Use the best state from SA
    if best_state is not None:
        x, y, r, w = best_state
        h = 2.0 - w
        
        # Local optimization: Gradient descent on the penalty function
        # Or just a few more SA steps with very low T
        # Let's do a simple hill climbing / gradient step
        # We can use scipy.optimize.minimize on the energy function
        # But energy is non-smooth due to max(0, ...).
        # However, it's differentiable almost everywhere.
        
        # Let's try a few iterations of "push apart" and "grow"
        # 1. Push apart overlapping circles
        for _ in range(100):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        # Push apart
                        if d < 1e-6:
                            dx = 0.01
                            dy = 0.01
                            d = np.sqrt(dx*dx + dy*dy)
                        push = (r_sum - d) / 2.0
                        nx = dx / d
                        ny = dy / d
                        x[i] += push * nx
                        y[i] += push * ny
                        x[j] -= push * nx
                        y[j] -= push * ny
                        moved = True
            
            # 2. Push against walls
            for i in range(N):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                if x[i] > w - r[i]:
                    x[i] = w - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                if y[i] > h - r[i]:
                    y[i] = h - r[i]
                    moved = True
            
            if not moved:
                break
        
        # 3. Grow radii
        # Try to increase each radius slightly
        for i in range(N):
            # Max radius allowed by walls
            r_max_wall = min(x[i], w - x[i], y[i], h - y[i])
            # Max radius allowed by neighbors
            r_max_neighbor = r_max_wall
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max_neighbor:
                    r_max_neighbor = r_allowed
            
            new_r = max(0.01, r_max_neighbor)
            if new_r > r[i]:
                r[i] = new_r
        
        # Check validity again
        valid, val = _check_and_get_value(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_state = (x.copy(), y.copy(), r.copy(), w)
            
    # Return best result
    if best_state is not None:
        x, y, r, w = best_state
        return np.column_stack((x, y, r))
    else:
        # Fallback: Grid
        side = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N: break
                cx = (i + 0.5) / 5.0
                cy = (j + 0.5) / 5.0
                r0 = 0.5 / 5.0 - 0.01
                cells.append([cx, cy, r0])
        return np.array(cells[:N])

def construct():
    return _construct()

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res)
