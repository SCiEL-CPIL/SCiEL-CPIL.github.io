"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return deadline - time.time()
    
    def compute_score(w, circles):
        """Compute the sum of radii, return 0 if invalid."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 0.0
        total_r = 0.0
        n = len(circles)
        for i in range(n):
            xi, yi, ri = circles[i]
            # Check inside rectangle
            if ri <= 0:
                return 0.0
            if xi - ri < -1e-10 or xi + ri > w + 1e-10:
                return 0.0
            if yi - ri < -1e-10 or yi + ri > h + 1e-10:
                return 0.0
            total_r += ri
        # Check no overlap
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < ri + rj - 1e-10:
                    return 0.0
        return total_r
    
    def penalty_score(w, circles):
        """Penalty-based score for optimization."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return -1e6
        penalty = 0.0
        total_r = 0.0
        n = len(circles)
        for i in range(n):
            xi, yi, ri = circles[i]
            if ri <= 0:
                penalty += 100.0
                continue
            total_r += ri
            # Inside rectangle penalty
            if xi - ri < 0:
                penalty += 100.0 * (xi - ri)
            if xi + ri > w:
                penalty += 100.0 * (xi + ri - w)
            if yi - ri < 0:
                penalty += 100.0 * (yi - ri)
            if yi + ri > h:
                penalty += 100.0 * (yi + ri - h)
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                if ri <= 0 or rj <= 0:
                    continue
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = ri + rj - dist
                if overlap > 0:
                    penalty += 100.0 * overlap
        return total_r - penalty
    
    def init_grid(w, h):
        """Initialize circles in a grid pattern."""
        # Try different grid configurations
        best_circles = None
        best_score = -1
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            cells = []
            for i in range(rows):
                for j in range(cols):
                    if len(cells) >= N:
                        break
                    cx = (j + 0.5) * w / cols
                    cy = (i + 0.5) * h / rows
                    r = min(w/cols, h/rows) * 0.49
                    cells.append([cx, cy, r])
            score = penalty_score(w, cells)
            if score > best_score:
                best_score = score
                best_circles = cells
        if best_circles is None:
            # Fallback: square grid
            side = int(np.ceil(np.sqrt(N)))
            best_circles = []
            for i in range(side):
                for j in range(side):
                    if len(best_circles) >= N:
                        break
                    cx = (j + 0.5) / side
                    cy = (i + 0.5) / side
                    r = 0.5 / side * 0.99
                    best_circles.append([cx, cy, r])
        return best_circles
    
    # Multi-start simulated annealing
    best_score = 0
    best_result = None
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    
    for ar in aspect_ratios:
        if elapsed() > deadline - 10:
            break
        
        w = ar / (1 + ar) * 2.0  # w + h = 2, w/h = ar => w = 2*ar/(1+ar)
        h = 2.0 - w
        
        circles = init_grid(w, h)
        current_score = penalty_score(w, circles)
        
        if compute_score(w, circles) > best_score:
            best_score = compute_score(w, circles)
            best_result = (w, circles[:])
        
        # Simulated annealing
        T = 0.1
        T_min = 0.001
        cooling = 0.995
        step_size = 0.05
        
        n_iter = 5000
        for it in range(n_iter):
            if elapsed() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Random perturbation
            new_w = w + random.gauss(0, 0.02)
            new_w = np.clip(new_w, 0.1, 1.9)
            new_circles = [list(c) for c in circles]
            
            # Perturb 1-3 random circles
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, N-1)
                new_circles[idx][0] += random.gauss(0, step_size)
                new_circles[idx][1] += random.gauss(0, step_size)
                new_circles[idx][2] += random.gauss(0, step_size * 0.5)
                new_circles[idx][2] = max(0.01, new_circles[idx][2])
            
            new_score = penalty_score(new_w, new_circles)
            
            # Acceptance criterion
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                w = new_w
                circles = new_circles
                current_score = new_score
                
                valid_score = compute_score(w, circles)
                if valid_score > best_score:
                    best_score = valid_score
                    best_result = (w, circles[:])
            
            T *= cooling
            step_size *= 0.999
        
        # Local polish for best result
        if elapsed() > deadline - 8:
            break
    
    # Final polish: try to increase radii for the best configuration
    if best_result is not None and elapsed() < deadline - 3:
        w, circles = best_result
        # Try small perturbations to improve
        for _ in range(100):
            if elapsed() > deadline - 2:
                break
            trial = [list(c) for c in circles]
            idx = random.randint(0, N-1)
            trial[idx][2] *= 1.01  # Try to increase radius
            if compute_score(w, trial) > best_score:
                best_score = compute_score(w, trial)
                best_result = (w, trial)
    
    # Build output
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w, circles = best_result
    result = np.array([[c[0], c[1], c[2]] for c in circles])
    return result

# EVOLVE-BLOCK-END
