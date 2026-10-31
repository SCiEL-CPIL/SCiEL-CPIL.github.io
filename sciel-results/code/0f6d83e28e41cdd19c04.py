"""Circle packing in perimeter-4 rectangle: pack 21 circles to maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, keep margin before 150s kill
    
    # Variables: [w, h, x_1..x_N, y_1..y_N, r_1..r_N]
    # Constraints: w + h = 2 (perimeter 4), circles in box, no overlap
    # Objective: maximize sum(r_i)
    
    def penalty_objective(vars, penalty_weight):
        w, h = vars[0], vars[1]
        centers = vars[2:2+2*N].reshape(N, 2)
        radii = vars[2+2*N:2+3*N]
        
        # Objective: -sum(radii)
        obj = -np.sum(radii)
        
        # Penalty for constraint violations
        pen = 0.0
        
        # Perimeter constraint: w + h = 2
        pen += penalty_weight * (w + h - 2.0)**2
        
        # Circles must be in rectangle
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if x - r < 0: pen += penalty_weight * (x - r)**2
            if x + r > w: pen += penalty_weight * (x + r - w)**2
            if y - r < 0: pen += penalty_weight * (y - r)**2
            if y + r > h: pen += penalty_weight * (y + r - h)**2
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    pen += penalty_weight * (min_dist - dist)**2
        
        return obj + pen
    
    def is_valid(vars, tol=1e-6):
        w, h = vars[0], vars[1]
        centers = vars[2:2+2*N].reshape(N, 2)
        radii = vars[2+2*N:2+3*N]
        
        if w + h > 2.0 + tol:
            return False
        if w <= 0 or h <= 0:
            return False
        
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if r < 0:
                return False
            if x - r < -tol or x + r > w + tol or y - r < -tol or y + r > h + tol:
                return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        
        return True
    
    def make_initial_config(w0, h0):
        """Create initial configuration with circles in a grid."""
        vars = np.zeros(2 + 3*N)
        vars[0] = w0
        vars[1] = h0
        
        # Place circles in a grid pattern
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        
        for i in range(N):
            col = i % cols
            row = i // cols
            x = (col + 0.5) * w0 / cols
            y = (row + 0.5) * h0 / rows
            r = min(w0 / cols, h0 / rows) * 0.45
            vars[2 + 2*i] = x
            vars[2 + 2*i + 1] = y
            vars[2 + 2*N + i] = r
        
        return vars
    
    def hexagonal_initial(w0, h0):
        """Create initial configuration with hexagonal packing."""
        vars = np.zeros(2 + 3*N)
        vars[0] = w0
        vars[1] = h0
        
        # Hexagonal packing
        r0 = min(w0, h0) / 8.0
        dx = 2.0 * r0
        dy = np.sqrt(3) * r0
        
        idx = 0
        row = 0
        while idx < N:
            col = 0
            while col < int(w0 / dx) + 2 and idx < N:
                x = (col + 0.5) * dx
                if row % 2 == 1:
                    x += dx / 2
                y = (row + 0.5) * dy
                if x < w0 and y < h0:
                    vars[2 + 2*idx] = x
                    vars[2 + 2*idx + 1] = y
                    vars[2 + 2*N + idx] = r0 * 0.95
                    idx += 1
                col += 1
            row += 1
        
        # Fill any remaining with small circles
        while idx < N:
            vars[2 + 2*idx] = w0 / 2
            vars[2 + 2*idx + 1] = h0 / 2
            vars[2 + 2*N + idx] = 0.01
            idx += 1
        
        return vars
    
    # Try multiple initial configurations and optimization strategies
    best_vars = None
    best_score = -np.inf
    
    initial_configs = [
        make_initial_config(1.0, 1.0),
        make_initial_config(0.8, 1.2),
        make_initial_config(1.2, 0.8),
        hexagonal_initial(1.0, 1.0),
        hexagonal_initial(0.9, 1.1),
    ]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(3):
        cfg = make_initial_config(1.0 + np.random.uniform(-0.2, 0.2), 1.0 + np.random.uniform(-0.2, 0.2))
        # Perturb positions and radii
        for i in range(N):
            cfg[2 + 2*i] += np.random.uniform(-0.05, 0.05)
            cfg[2 + 2*i + 1] += np.random.uniform(-0.05, 0.05)
            cfg[2 + 2*N + i] *= (1 + np.random.uniform(-0.2, 0.2))
        initial_configs.append(cfg)
    
    for init_idx, init_vars in enumerate(initial_configs):
        if time.time() - start_time > deadline - 30:
            break
        
        # Use penalty method with increasing penalty weight
        penalty_weights = [10, 100, 1000, 10000]
        current_vars = init_vars.copy()
        
        for pw in penalty_weights:
            if time.time() - start_time > deadline - 20:
                break
            
            try:
                res = minimize(
                    penalty_objective,
                    current_vars,
                    args=(pw,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                current_vars = res.x
            except Exception:
                pass
        
        # Check if valid
        if is_valid(current_vars):
            score = np.sum(current_vars[2+2*N:2+3*N])
            if score > best_score:
                best_score = score
                best_vars = current_vars.copy()
    
    # If no valid solution found, fall back to a simple valid configuration
    if best_vars is None:
        # Simple grid in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        result = np.array(cells[:N])
        return result
    
    # Extract result: [x_i, y_i, r_i] for each circle
    w, h = best_vars[0], best_vars[1]
    centers = best_vars[2:2+2*N].reshape(N, 2)
    radii = best_vars[2+2*N:2+3*N]
    
    # Clamp to ensure validity
    result = np.zeros((N, 3))
    for i in range(N):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # Clamp radius
        r = max(0, min(r, min(x, w-x, y, h-y)))
        # Clamp position
        x = max(r, min(w - r, x))
        y = max(r, min(h - r, y))
        result[i] = [x, y, r]
    
    # Final validation and repair
    for iter in range(50):
        if time.time() - start_time > deadline - 5:
            break
        
        # Check for violations and reduce radii if needed
        valid = True
        for i in range(N):
            x, y, r = result[i]
            if x - r < 0 or x + r > w or y - r < 0 or y + r > h:
                valid = False
                # Reduce radius to fit in box
                max_r = min(x, w-x, y, h-y)
                if r > max_r:
                    result[i, 2] = max(0, max_r)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result[i, 2] + result[j, 2]
                if dist < min_dist - 1e-9:
                    valid = False
                    # Reduce radii to avoid overlap
                    scale = dist / (min_dist + 1e-9)
                    result[i, 2] *= scale
                    result[j, 2] *= scale
        
        if valid:
            break
    
    return result
