"""
Strategy: Simulated Annealing for Circle Packing
1. Jointly optimize circle positions, radii, and rectangle aspect ratio.
2. Use a soft penalty energy function to handle constraints.
3. Perform a final polish pass to ensure validity and maximize sum of radii.
"""

import numpy as np
import time
import random
import math

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Pre-compute pairwise indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.stack([i_idx, j_idx], axis=1)
    
    def get_dimensions(aspect):
        """Calculate width and height from aspect ratio, given perimeter 4."""
        # w + h = 2
        # w / h = aspect
        # w = aspect * h
        # aspect * h + h = 2 => h = 2 / (1 + aspect)
        # w = 2 * aspect / (1 + aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def calculate_energy(x, aspect, penalty_weight=1000.0):
        """
        x: array of shape (3*N) -> [x1, y1, r1, ..., xN, yN, rN]
        aspect: float
        Returns: Energy (lower is better). 
        Energy = penalty_weight * (sum of violations) - sum(r)
        """
        w, h = get_dimensions(aspect)
        
        xy = x[0::3]
        yv = x[1::3]
        r = x[2::3]
        
        # 1. Boundary Violations
        # Circle i must satisfy: r_i <= x_i <= w - r_i and r_i <= y_i <= h - r_i
        # Violation is max(0, r_i - x_i) + max(0, x_i - (w - r_i)) + ...
        
        v_x_min = np.maximum(0, r - xy)
        v_x_max = np.maximum(0, xy - (w - r))
        v_y_min = np.maximum(0, r - yv)
        v_y_max = np.maximum(0, yv - (h - r))
        
        boundary_violation = np.sum(v_x_min) + np.sum(v_x_max) + np.sum(v_y_min) + np.sum(v_y_max)
        
        # 2. Overlap Violations
        # dist(i, j) >= r_i + r_j
        # Violation is max(0, r_i + r_j - dist(i, j))
        
        dx = xy[pairs[:, 0]] - xy[pairs[:, 1]]
        dy = yv[pairs[:, 0]] - yv[pairs[:, 1]]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        
        r_sum = r[pairs[:, 0]] + r[pairs[:, 1]]
        overlap_violation = np.sum(np.maximum(0, r_sum - dist))
        
        total_violation = boundary_violation + overlap_violation
        
        # Objective: Minimize penalty - sum(r)
        # We want to maximize sum(r). So subtract it.
        energy = penalty_weight * total_violation - np.sum(r)
        
        return energy

    def is_valid(x, aspect, tol=1e-5):
        w, h = get_dimensions(aspect)
        xy = x[0::3]
        yv = x[1::3]
        r = x[2::3]
        
        # Check boundaries
        if np.any(xy < r - tol) or np.any(xy > w - r + tol) or \
           np.any(yv < r - tol) or np.any(yv > h - r + tol):
            return False
            
        # Check overlaps
        dx = xy[pairs[:, 0]] - xy[pairs[:, 1]]
        dy = yv[pairs[:, 0]] - yv[pairs[:, 1]]
        dist = np.sqrt(dx*dx + dy*dy)
        r_sum = r[pairs[:, 0]] + r[pairs[:, 1]]
        
        if np.any(dist < r_sum - tol):
            return False
            
        return True

    def get_initial_state(aspect):
        w, h = get_dimensions(aspect)
        # Estimate radius
        r_approx = np.sqrt(w * h / (N * np.pi)) * 0.8
        
        x = np.zeros(3 * N)
        # Hexagonal-ish grid initialization
        # Try to fit N circles in a grid
        cols = int(np.ceil(math.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        
        # Adjust cols/rows to be closer to N
        # Simple loop to place them
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                # Center of cell
                cx = (col + 0.5) * (w / cols)
                cy = (row + 0.5) * (h / rows)
                
                # Jitter
                cx += np.random.uniform(-0.1, 0.1) * (w/cols)
                cy += np.random.uniform(-0.1, 0.1) * (h/rows)
                
                # Clamp
                cx = np.clip(cx, r_approx, w - r_approx)
                cy = np.clip(cy, r_approx, h - r_approx)
                
                x[idx*3] = cx
                x[idx*3+1] = cy
                x[idx*3+2] = r_approx
                idx += 1
        
        return x

    best_energy = np.inf
    best_x = None
    best_aspect = 1.0
    best_valid_score = 0.0
    best_valid_result = None
    
    # Multi-start SA
    aspects_to_try = [0.8, 1.0, 1.25, 1.5]
    random.seed(42)
    np.random.seed(42)
    
    for asp_init in aspects_to_try:
        if time.time() > deadline - 10:
            break
            
        x = get_initial_state(asp_init)
        aspect = asp_init
        
        # SA Parameters
        T = 1.0
        T_min = 1e-6
        cooling = 0.995
        max_iter = 50000
        penalty_weight = 1000.0
        
        current_energy = calculate_energy(x, aspect, penalty_weight)
        
        # Track best valid solution during SA
        # We check validity occasionally to save time, or at the end of each cooling step?
        # Checking every 1000 steps is efficient enough.
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Generate new state
            x_new = x.copy()
            aspect_new = aspect
            
            move_type = random.random()
            if move_type < 0.7:
                # Move a circle
                idx = random.randint(0, N - 1)
                # Random walk in position
                step = 0.05 * T / 0.1 # Scale step with T? Or fixed?
                # Let's use a fixed scale for positions, scaled by T for radii
                x_new[idx*3] += np.random.normal(0, 0.1)
                x_new[idx*3+1] += np.random.normal(0, 0.1)
                
                # Clamp to large bounds to avoid infinity
                w, h = get_dimensions(aspect)
                x_new[idx*3] = np.clip(x_new[idx*3], -1, w+1)
                x_new[idx*3+1] = np.clip(x_new[idx*3+1], -1, h+1)
                
            elif move_type < 0.9:
                # Change radius
                idx = random.randint(0, N - 1)
                x_new[idx*3+2] += np.random.normal(0, 0.05)
                if x_new[idx*3+2] < 0.01:
                    x_new[idx*3+2] = 0.01
            else:
                # Change aspect ratio
                aspect_new = aspect * (1 + np.random.normal(0, 0.1))
                # Keep aspect in reasonable range [0.5, 2.0]
                aspect_new = np.clip(aspect_new, 0.5, 2.0)
                
            new_energy = calculate_energy(x_new, aspect_new, penalty_weight)
            
            # Acceptance
            delta_E = new_energy - current_energy
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                x = x_new
                aspect = aspect_new
                current_energy = new_energy
                
                # Check for valid solution periodically
                if i % 1000 == 0:
                    if is_valid(x, aspect):
                        score = np.sum(x[2::3])
                        if score > best_valid_score:
                            best_valid_score = score
                            best_valid_result = np.column_stack((x[0::3], x[1::3], x[2::3])).copy()
                            best_aspect = aspect
                            best_x = x.copy()
            
            T *= cooling
            if T < T_min:
                break
        
        # Final check for this start
        if is_valid(x, aspect):
            score = np.sum(x[2::3])
            if score > best_valid_score:
                best_valid_score = score
                best_valid_result = np.column_stack((x[0::3], x[1::3], x[2::3])).copy()
                best_aspect = aspect
                best_x = x.copy()

    # Polish Phase
    if best_valid_result is not None and time.time() < deadline - 20:
        # Use the best found state to run a short, high-precision SA or local search
        # We will run SA again starting from best_x with lower T and more iterations
        # to squeeze out more value.
        
        x_polish = best_x.copy()
        aspect_polish = best_aspect
        T = 0.1
        T_min = 1e-8
        cooling = 0.99
        max_iter_polish = 100000
        penalty_weight = 1000.0
        
        current_energy = calculate_energy(x_polish, aspect_polish, penalty_weight)
        
        for i in range(max_iter_polish):
            if time.time() > deadline - 5:
                break
                
            x_new = x_polish.copy()
            aspect_new = aspect_polish
            
            move_type = random.random()
            if move_type < 0.7:
                idx = random.randint(0, N - 1)
                # Smaller steps
                x_new[idx*3] += np.random.normal(0, 0.01)
                x_new[idx*3+1] += np.random.normal(0, 0.01)
                w, h = get_dimensions(aspect_polish)
                x_new[idx*3] = np.clip(x_new[idx*3], 0, w)
                x_new[idx*3+1] = np.clip(x_new[idx*3+1], 0, h)
            elif move_type < 0.9:
                idx = random.randint(0, N - 1)
                x_new[idx*3+2] += np.random.normal(0, 0.01)
                if x_new[idx*3+2] < 0.01:
                    x_new[idx*3+2] = 0.01
            else:
                aspect_new = aspect_polish * (1 + np.random.normal(0, 0.05))
                aspect_new = np.clip(aspect_new, 0.5, 2.0)
                
            new_energy = calculate_energy(x_new, aspect_new, penalty_weight)
            
            delta_E = new_energy - current_energy
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                x_polish = x_new
                aspect_polish = aspect_new
                current_energy = new_energy
                
                if i % 500 == 0:
                    if is_valid(x_polish, aspect_polish):
                        score = np.sum(x_polish[2::3])
                        if score > best_valid_score:
                            best_valid_score = score
                            best_valid_result = np.column_stack((x_polish[0::3], x_polish[1::3], x_polish[2::3])).copy()
                            best_aspect = aspect_polish
                            best_x = x_polish.copy()
            
            T *= cooling
            if T < T_min:
                break

    if best_valid_result is not None:
        return best_valid_result
    else:
        # Fallback: Grid of small circles
        res = np.zeros((N, 3))
        w, h = 1.0, 1.0
        side = 5
        r = 0.08
        for i in range(N):
            row = i // side
            col = i % side
            res[i, 0] = (col + 0.5) * (w/side)
            res[i, 1] = (row + 0.5) * (h/side)
            res[i, 2] = r
        return res

# EVOLVE-BLOCK-END
