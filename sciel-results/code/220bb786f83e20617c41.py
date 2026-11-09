"""Rewritten program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    def validate(positions, radii, w, h, tol=1e-8):
        """Check if circles are valid (inside rectangle, no overlaps)."""
        for i in range(N):
            if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
                return False
            if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
                return False
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def compute_radii(positions, w, h, iters=200):
        """Iteratively compute maximum radii for given positions."""
        radii = np.zeros(N)
        for i in range(N):
            radii[i] = min(positions[i, 0], w - positions[i, 0], 
                          positions[i, 1], h - positions[i, 1]) * 0.1
        radii = np.maximum(radii, 0.001)
        
        for _ in range(iters):
            new_radii = np.zeros(N)
            for i in range(N):
                max_r = min(positions[i, 0], w - positions[i, 0], 
                           positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i == j:
                        continue
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    avail = dist - radii[j]
                    if avail < max_r:
                        max_r = avail
                new_radii[i] = max(0.0, max_r)
            radii = 0.5 * radii + 0.5 * new_radii
        
        return radii
    
    def solve_for_aspect(ar):
        """Solve the packing problem for a given aspect ratio."""
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Initial grid-based positions
        cols = max(1, int(round(np.sqrt(N * ar))))
        rows = max(1, int(np.ceil(N / cols)))
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            cell_w = w / cols
            cell_h = h / rows
            positions[i, 0] = (col + 0.5) * cell_w
            positions[i, 1] = (row + 0.5) * cell_h
        
        # Compute initial radii
        radii = compute_radii(positions, w, h, iters=100)
        
        # Optimize positions to maximize sum of radii
        def objective(params):
            pos = params.reshape(N, 2)
            r = compute_radii(pos, w, h, iters=50)
            return -np.sum(r)
        
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
        
        try:
            res = minimize(objective, positions.flatten(), method='SLSQP',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success:
                new_positions = res.x.reshape(N, 2)
                new_radii = compute_radii(new_positions, w, h, iters=200)
                
                if validate(new_positions, new_radii, w, h):
                    return new_positions, new_radii, np.sum(new_radii)
        except Exception:
            pass
        
        if validate(positions, radii, w, h):
            return positions, radii, np.sum(radii)
        
        return None, None, -1.0
    
    best_result = None
    best_score = -1.0
    
    # Sweep aspect ratios
    aspect_ratios = np.linspace(0.3, 3.3, 20)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        time_budget = deadline - time.time()
        if time_budget < 5:
            break
        
        positions, radii, score = solve_for_aspect(ar)
        
        if positions is not None and score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()
    
    # Fallback if no valid result
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
