import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety

    def unpack(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return x, y, r

    def pack(x, y, r):
        return np.column_stack((x, y, r)).flatten()

    def get_score_and_validity(params):
        x, y, r = unpack(params)
        
        # Check validity
        valid = True
        # Boundary check
        if np.any(x < r - 1e-7) or np.any(x > 1 - r + 1e-7):
            valid = False
        if np.any(y < r - 1e-7) or np.any(y > 1 - r + 1e-7):
            valid = False
            
        # Overlap check
        if valid:
            diff_x = x[:, None] - x[None, :]
            diff_y = y[:, None] - y[None, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            min_dist = r[:, None] + r[None, :]
            np.fill_diagonal(min_dist, 0)
            if np.any(dist < min_dist - 1e-7):
                valid = False
        
        score = np.sum(r) if valid else -np.inf
        return score, valid

    def objective(params):
        x, y, r = unpack(params)
        
        # Boundary penalties
        # x in [r, 1-r]
        pen_x_low = np.maximum(0, r - x)**2
        pen_x_high = np.maximum(0, x - (1 - r))**2
        pen_y_low = np.maximum(0, r - y)**2
        pen_y_high = np.maximum(0, y - (1 - r))**2
        boundary_pen = np.sum(pen_x_low + pen_x_high + pen_y_low + pen_y_high)
        
        # Overlap penalties
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        overlap_pen = np.sum(viol**2)
        
        total_pen = boundary_pen + overlap_pen
        sum_r = np.sum(r)
        
        # Strong penalty to ensure validity, but allow some flexibility during optimization
        return -sum_r + 100.0 * total_pen

    def force_directed_step(params, dt=0.01, iterations=100):
        """
        Performs force-directed relaxation to resolve overlaps and push circles into corners/edges.
        Returns updated params.
        """
        x, y, r = unpack(params)
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        k_repel = 50.0
        k_wall = 50.0
        damping = 0.9
        
        for _ in range(iterations):
            if time.time() > deadline - 10:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-8:
                        # Force magnitude proportional to overlap
                        f = k_repel * (min_d - d)
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f
                    elif d <= 1e-8:
                        # Random push if exactly overlapping
                        fx[i] += np.random.uniform(-1, 1)
                        fy[i] += np.random.uniform(-1, 1)
                        fx[j] -= np.random.uniform(-1, 1)
                        fy[j] -= np.random.uniform(-1, 1)
            
            # Wall repulsion
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += k_wall * (r[i] - x[i])
                # Right wall
                if x[i] > 1 - r[i]:
                    fx[i] -= k_wall * (x[i] - (1 - r[i]))
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += k_wall * (r[i] - y[i])
                # Top wall
                if y[i] > 1 - r[i]:
                    fy[i] -= k_wall * (y[i] - (1 - r[i]))
            
            # Update velocities and positions
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            
            x += vx * dt
            y += vy * dt
            
            # Hard clip to prevent flying out of bounds (soft penalty handles the rest)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Ensure radii don't shrink below a minimum in this phase (we only move positions)
            
        return pack(x, y, r)

    def get_initial_guesses():
        guesses = []
        
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([
                        (i + 0.5) / side,
                        (j + 0.5) / side,
                        0.5 / side * 0.95
                    ])
        guesses.append(np.array(cells[:N]).flatten())
        
        # 2. Random with small radii
        np.random.seed(42)
        for _ in range(3):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.05 + 0.02
            guesses.append(pack(x, y, r))
            
        # 3. Hexagonal packing approximation
        # This is a heuristic for denser packing
        h = 0.2
        w = h * np.sqrt(3) / 2
        hex_cells = []
        for row in range(5):
            for col in range(5):
                if len(hex_cells) >= N: break
                y_pos = 0.1 + row * w
                x_offset = 0.1 if row % 2 == 0 else 0.1 + h/2
                x_pos = x_offset + col * h
                if x_pos < 0.9 and y_pos < 0.9:
                    hex_cells.append([x_pos, y_pos, 0.1])
        if len(hex_cells) >= N:
            guesses.append(np.array(hex_cells[:N]).flatten())
            
        return guesses

    best_result = None
    best_score = -np.inf
    
    guesses = get_initial_guesses()
    
    # Phase 1: Force-Directed Search
    # We use force-directed steps to explore the landscape.
    # We interleave with local optimization to refine.
    
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 20:
            break
            
        current_params = guess.copy()
        
        # Run a few iterations of force-directed relaxation
        # We do this in chunks to allow checking time and potentially polishing
        for step in range(5):
            if time.time() > deadline - 10:
                break
                
            # Force directed move
            current_params = force_directed_step(current_params, dt=0.05, iterations=50)
            
            # Check score after force step
            score, valid = get_score_and_validity(current_params)
            if valid and score > best_score:
                best_score = score
                best_result = current_params.copy()
                
            # If we have time, do a quick local polish
            if time.time() < deadline - 15:
                try:
                    res = minimize(objective, current_params, method='L-BFGS-B', 
                                   options={'maxiter': 50, 'ftol': 1e-6})
                    if res.fun < objective(current_params): # Only accept if improved
                        candidate = res.x
                        c_score, c_valid = get_score_and_validity(candidate)
                        if c_valid and c_score > best_score:
                            best_score = c_score
                            best_result = candidate.copy()
                            current_params = candidate.copy()
                except Exception:
                    pass

    # Phase 2: Final Polish on the best result found
    if best_result is not None:
        if time.time() < deadline - 5:
            try:
                # More aggressive local optimization
                res = minimize(objective, best_result, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-8})
                candidate = res.x
                c_score, c_valid = get_score_and_validity(candidate)
                if c_valid and c_score > best_score:
                    best_score = c_score
                    best_result = candidate
            except Exception:
                pass

    # Fallback if no valid result
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final Safety Repair
    x, y, r = unpack(best_result)
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # Iterative shrink repair for any remaining overlaps due to floating point or aggressive clipping
    for _ in range(20):
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        
        if np.max(viol) < 1e-8:
            break
            
        idx_i, idx_j = np.unravel_index(np.argmax(viol), viol.shape)
        reduction = viol[idx_i, idx_j] / 2 + 1e-7
        r[idx_i] = max(r[idx_i] - reduction, 1e-6)
        r[idx_j] = max(r[idx_j] - reduction, 1e-6)
        
        # Re-clip positions if radii changed significantly? 
        # Since we only shrink, positions remain valid for boundaries if they were before.
        
    return pack(x, y, r)
