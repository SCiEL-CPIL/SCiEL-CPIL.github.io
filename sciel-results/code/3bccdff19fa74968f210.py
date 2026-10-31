N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -np.inf
    best_w = 1.0
    best_h = 1.0

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score, best_w, best_h
        if validate(w, h, x, y, r):
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                best_w = w
                best_h = h
                return True
        return False

    def energy(w, h, x, y, r):
        """
        Calculate energy: Penalties for overlaps and wall violations, minus reward for radii.
        """
        e = 0.0
        # Wall penalties
        for i in range(N):
            if x[i] < r[i]:
                e += (r[i] - x[i]) ** 2
            if w - x[i] < r[i]:
                e += (r[i] - (w - x[i])) ** 2
            if y[i] < r[i]:
                e += (r[i] - y[i]) ** 2
            if h - y[i] < r[i]:
                e += (r[i] - (h - y[i])) ** 2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    e += overlap ** 2 * 10.0 # Strong penalty for overlaps
        
        # Reward for radii
        e -= np.sum(r) * 5.0
        
        return e

    def simulated_annealing(w, h, x0, y0, r0, max_iters=5000, T0=0.05, T_min=0.001, cooling=0.999):
        """
        Run Simulated Annealing to optimize the packing.
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        # Initial energy
        E_curr = energy(w, h, x, y, r)
        
        # Track best valid solution found during SA
        best_local_x = x.copy()
        best_local_y = y.copy()
        best_local_r = r.copy()
        best_local_score = -np.inf
        
        # Check if initial is valid
        if validate(w, h, x, y, r):
            best_local_score = np.sum(r)
        
        T = T0
        step_size = 0.1
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Generate move
            idx = np.random.randint(N)
            
            # Decide which parameter to change
            move_type = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            
            if move_type == 0:
                # Move x
                x_new[idx] += np.random.normal(0, step_size)
            elif move_type == 1:
                # Move y
                y_new[idx] += np.random.normal(0, step_size)
            else:
                # Change radius
                r_new[idx] += np.random.normal(0, step_size * 0.5)
                r_new[idx] = max(0.01, r_new[idx]) # Keep radius positive
                
            # Clip positions to box (soft constraint handled by energy, but hard clip helps)
            # Actually, let energy handle walls to allow exploration, but keep them reasonable
            # Hard clipping might prevent escaping local minima if we clip too aggressively
            # Let's just calculate energy.
            
            E_new = energy(w, h, x_new, y_new, r_new)
            
            # Acceptance probability
            dE = E_new - E_curr
            if dE < 0 or np.random.random() < np.exp(-dE / T):
                x = x_new
                y = y_new
                r = r_new
                E_curr = E_new
                
                # Check validity for best local
                if validate(w, h, x, y, r):
                    score = np.sum(r)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_x = x.copy()
                        best_local_y = y.copy()
                        best_local_r = r.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Slowly decrease step size
            if i % 100 == 0:
                step_size *= 0.95
                
        return best_local_x, best_local_y, best_local_r

    # Main search loop
    np.random.seed(42)
    
    # Try different aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        # Calculate w, h from aspect ratio
        # w/h = aspect, w + h = 2 => w = 2*aspect/(1+aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        # Initialize
        # Grid init is good for SA to start from a reasonable place
        n_cols = int(np.round(np.sqrt(N * w/h)))
        n_rows = int(np.ceil(N / n_cols))
        while n_cols * n_rows < N:
            n_cols += 1
            
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        max_r = min(cell_w, cell_h) * 0.45
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = max_r
                idx += 1
        
        # Run SA
        # Adjust max_iters based on time remaining
        time_left = deadline - time.time()
        # Estimate iterations per second (rough)
        # Just use a fixed high number, SA loop has time check
        x_best, y_best, r_best = simulated_annealing(w, h, x, y, r, max_iters=20000)
        
        update_best(w, h, x_best, y_best, r_best)
        
        # Try a random initialization as well
        if time.time() < deadline - 10.0:
            x_rand = np.random.uniform(r, w-r, N)
            y_rand = np.random.uniform(r, h-r, N)
            r_rand = np.random.uniform(0.05, 0.15, N)
            
            x_rand2, y_rand2, r_rand2 = simulated_annealing(w, h, x_rand, y_rand, r_rand, max_iters=10000)
            update_best(w, h, x_rand2, y_rand2, r_rand2)

    # Final Polish: Local optimization on the best result found
    if best_result is not None and time.time() < deadline - 2.0:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        w = best_w
        h = best_h
        
        # Short SA run to polish
        x_polished, y_polished, r_polished = simulated_annealing(w, h, x, y, r, max_iters=5000, T0=0.01, T_min=0.001, cooling=0.999)
        update_best(w, h, x_polished, y_polished, r_polished)

    # Fallback
    if best_result is None:
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Simple grid
        w, h = 1.0, 1.0
        n_cols = 5
        n_rows = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            x[i] = (col + 0.5) * 0.2
            y[i] = (row + 0.5) * 0.2
            r[i] = 0.09
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
