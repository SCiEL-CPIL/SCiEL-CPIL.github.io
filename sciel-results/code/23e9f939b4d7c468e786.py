import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_dims(t):
        t = np.clip(t, 0.15, 0.85)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h
    
    def compute_penalty(t, circles, mu=10.0):
        """
        Computes the penalty for the given configuration.
        Penalty = sum of squared overlaps + sum of squared boundary violations.
        """
        w, h = get_dims(t)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Boundary violations
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x - (w - r))
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y - (h - r))
        
        boundary_penalty = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r**2:
                    dist = np.sqrt(dist_sq)
                    overlap = sum_r - dist
                    overlap_penalty += overlap**2
        
        return boundary_penalty + overlap_penalty

    def evaluate(t, circles):
        """
        Returns (score, penalty, is_valid)
        """
        penalty = compute_penalty(t, circles)
        score = np.sum(circles[:, 2])
        is_valid = penalty < 1e-7
        return score, penalty, is_valid

    def initial_config(t):
        """Generate a valid initial configuration using grid placement."""
        w, h = get_dims(t)
        circles = np.zeros((N, 3))
        
        # Use a grid pattern that fits 21 circles
        # Try different grid layouts
        layouts = [
            (5, 5),  # 5x5 grid, use 21
            (4, 6),  # 4x6 grid, use 24, take 21
            (3, 7),  # 3x7 grid, use 21
            (7, 3),  # 7x3 grid
            (6, 4),  # 6x4 grid
        ]
        
        best_circles = None
        best_score = -np.inf
        
        for rows, cols in layouts:
            if rows * cols < N:
                continue
            
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles = np.zeros((N, 3))
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= N:
                        break
                    cx = (j + 0.5) * cell_w
                    cy = (i + 0.5) * cell_h
                    circles[count, 0] = cx
                    circles[count, 1] = cy
                    circles[count, 2] = r
                    count += 1
            
            score, penalty, is_valid = evaluate(t, circles)
            if is_valid and score > best_score:
                best_score = score
                best_circles = circles.copy()
        
        if best_circles is None:
            # Fallback: very small circles in a 5x5 grid
            w, h = get_dims(t)
            circles = np.zeros((N, 3))
            count = 0
            for i in range(5):
                for j in range(5):
                    if count >= N:
                        break
                    cx = (j + 0.5) * (w / 5)
                    cy = (i + 0.5) * (h / 5)
                    r = 0.3 * min(w, h) / 5
                    circles[count, 0] = cx
                    circles[count, 1] = cy
                    circles[count, 2] = r
                    count += 1
            best_circles = circles
        
        return best_circles

    def simulated_annealing(t_init, circles_init, max_iters=5000):
        """Simulated annealing to optimize circle packing."""
        t = t_init
        circles = circles_init.copy()
        
        best_score = -np.inf
        best_t = t
        best_circles = circles.copy()
        
        # Initial temperature
        T = 0.01
        cooling_rate = 0.9995
        
        current_score, current_penalty, _ = evaluate(t, circles)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate perturbation
            new_t = t
            new_circles = circles.copy()
            
            # Perturb aspect ratio
            if random.random() < 0.1:
                new_t = np.clip(t + random.gauss(0, 0.02), 0.15, 0.85)
            
            # Perturb one or more circles
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, N - 1)
                w, h = get_dims(new_t)
                
                # Perturb position
                new_circles[idx, 0] += random.gauss(0, 0.05)
                new_circles[idx, 1] += random.gauss(0, 0.05)
                
                # Perturb radius
                new_circles[idx, 2] *= (1 + random.gauss(0, 0.1))
                new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, min(w, h) / 2)
                
                # Keep within bounds
                new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, w)
                new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, h)
            
            new_score, new_penalty, _ = evaluate(new_t, new_circles)
            
            # Acceptance criterion
            delta_score = new_score - current_score
            if delta_score > 0 or random.random() < np.exp(delta_score / T):
                t = new_t
                circles = new_circles
                current_score = new_score
                current_penalty = new_penalty
                
                if new_penalty < 1e-7 and new_score > best_score:
                    best_score = new_score
                    best_t = t
                    best_circles = circles.copy()
            
            # Cool down
            T *= cooling_rate
        
        return best_t, best_circles, best_score

    best_score = -np.inf
    best_t = 0.5
    best_circles = None
    
    # Try multiple initial aspect ratios with simulated annealing
    random.seed(42)
    np.random.seed(42)
    
    t_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for t in t_values:
        if time.time() > deadline - 15:
            break
        
        circles_init = initial_config(t)
        
        t_opt, circles_opt, score_opt = simulated_annealing(t, circles_init, max_iters=3000)
        
        if score_opt > best_score:
            best_score = score_opt
            best_t = t_opt
            best_circles = circles_opt
    
    # If we didn't find a valid solution, fallback to initial grid
    if best_circles is None or best_score < 0.5:
        t = 0.5
        circles = initial_config(t)
        score, penalty, is_valid = evaluate(t, circles)
        if is_valid:
            best_score = score
            best_t = t
            best_circles = circles

    return best_circles
