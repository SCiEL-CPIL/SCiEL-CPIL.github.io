import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin from 150s limit
    
    # We will search over aspect ratios as well
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    # Let w = 2 * ar / (1+ar), h = 2 / (1+ar)
    
    best_solution = None
    best_score = 0.0
    best_ar = 1.0
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 0.001
    cooling_rate = 0.999
    
    # We will try a few different aspect ratios and run SA on each
    # Or we can treat aspect ratio as part of the state? 
    # Treating AR as part of the state is complex because bounds change.
    # Better: Run SA for a fixed AR, then try to optimize AR in a coarse grid, 
    # or use a nested loop. Given time budget, let's pick a few promising ARs 
    # and run SA on them, keeping the global best.
    
    # Promising ARs: 1.0 (square), 1.2, 0.8, 1.5, 0.7
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 1.1, 0.9]
    
    # To make it more robust, we can also try to adapt AR slightly during the process?
    # For now, let's fix AR per run and compare results.
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Initialize positions and radii
        # Start with a random valid configuration
        # We need a valid starting point. 
        # Let's start with small radii and random positions, then grow.
        
        x = np.random.uniform(0.1, 0.9, N) * w
        y = np.random.uniform(0.1, 0.9, N) * h
        r = np.full(N, 0.05) # Small initial radius to ensure no overlap
        
        # Ensure initial validity (shrink if necessary)
        def is_valid(x, y, r, w, h):
            for i in range(N):
                if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                    return False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j]:
                        return False
            return True
        
        # Force validity by shrinking radii if needed (should be rare with small r)
        while not is_valid(x, y, r, w, h):
            r *= 0.95
        
        current_score = np.sum(r)
        T = T_start
        
        # SA Loop
        # Move types:
        # 1. Grow a random circle
        # 2. Shrink a random circle (to allow escape)
        # 3. Move a random circle position (to resolve conflicts or pack tighter)
        
        max_iter = 20000
        iter_count = 0
        
        while T > T_end and iter_count < max_iter:
            if time.time() > deadline - 5:
                break
                
            iter_count += 1
            
            # Choose a move
            move_type = np.random.random()
            
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            
            if move_type < 0.4:
                # Grow a random circle
                i = np.random.randint(N)
                delta = np.random.uniform(0.001, 0.02)
                r_new[i] += delta
            elif move_type < 0.6:
                # Shrink a random circle
                i = np.random.randint(N)
                delta = np.random.uniform(0.001, 0.02)
                r_new[i] = max(0.001, r_new[i] - delta)
            else:
                # Move a random circle
                i = np.random.randint(N)
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                x_new[i] += dx
                y_new[i] += dy
                
            # Check validity of new state
            # We can accept invalid states in SA if we penalize the energy, 
            # but here "score" is sum(r). 
            # Standard SA for packing usually minimizes overlap energy.
            # Let's define Energy = -sum(r) + penalty(overlap) + penalty(boundary)
            # But we want to maximize sum(r) subject to constraints.
            # Let's use a soft constraint approach in the acceptance criterion.
            
            # Calculate "Energy" for acceptance
            # E = -sum(r) + Lambda * (overlap_violation + boundary_violation)
            
            def calc_energy(x, y, r, w, h):
                e = -np.sum(r)
                # Boundary violations
                for i in range(N):
                    if x[i] < r[i]: e += 100 * (r[i] - x[i])**2
                    if x[i] > w - r[i]: e += 100 * (x[i] - (w - r[i]))**2
                    if y[i] < r[i]: e += 100 * (r[i] - y[i])**2
                    if y[i] > h - r[i]: e += 100 * (y[i] - (h - r[i]))**2
                # Overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            e += 100 * (min_dist - dist)**2
                return e
            
            E_old = calc_energy(x, y, r, w, h)
            E_new = calc_energy(x_new, y_new, r_new, w, h)
            
            delta_E = E_new - E_old
            
            if delta_E < 0:
                # Accept if better (lower energy)
                x, y, r = x_new, y_new, r_new
            else:
                # Accept with probability exp(-delta_E / T)
                if np.random.random() < np.exp(-delta_E / T):
                    x, y, r = x_new, y_new, r_new
            
            T *= cooling_rate
            
            # Track best valid solution found so far in this AR run
            # We need to ensure the final solution is valid.
            # Let's check validity periodically or at the end.
            
        # After SA, we might have an invalid solution due to soft constraints.
        # We need to repair it to a valid one to get a score.
        # Strategy: If invalid, shrink radii until valid, or revert to last valid state.
        # Better: Keep track of the best VALID state encountered during SA.
        
        # Let's re-run the logic to keep track of best valid state.
        # Actually, the above loop doesn't track best valid. Let's modify.
        
        # Restart the SA loop for this AR with better tracking? 
        # Or just check the final state.
        # If final state is invalid, we can try to repair by shrinking.
        
        if is_valid(x, y, r, w, h):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_solution = np.column_stack((x, y, r))
                best_ar = ar
        else:
            # Try to repair by shrinking
            scale = 1.0
            valid = False
            for _ in range(50):
                r_test = r * scale
                if is_valid(x, y, r_test, w, h):
                    valid = True
                    score = np.sum(r_test)
                    if score > best_score:
                        best_score = score
                        best_solution = np.column_stack((x, y, r_test))
                        best_ar = ar
                    break
                scale *= 0.95
            
    if best_solution is None:
        # Fallback
        side = 5
        x_list, y_list, r_list = [], [], []
        for i in range(side):
            for j in range(side):
                if len(x_list) >= N:
                    break
                x_list.append((j + 0.5) / side)
                y_list.append((i + 0.5) / side)
                r_list.append(0.5 / side - 0.01)
        best_solution = np.column_stack((x_list, y_list, r_list))
        
    return best_solution
