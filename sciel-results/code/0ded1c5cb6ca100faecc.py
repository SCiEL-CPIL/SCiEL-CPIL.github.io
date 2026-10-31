"""Circle packing in perimeter-4 rectangle using contact-graph + simulated annealing."""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_value = 0.0
    best_result = None
    
    # Search over aspect ratios
    aspect_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w + h = 2, h/w = ar => w(1+ar) = 2 => w = 2/(1+ar)
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Try multiple random initializations for this aspect ratio
        for init_seed in range(3):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(hash((ar, init_seed)) % (2**32))
            
            # Generate initial configuration
            centers, radii = _generate_initial(w, h, rng)
            
            # Run simulated annealing
            result = _simulated_annealing(centers, radii, w, h, rng, 
                                         time_budget=min(10.0, deadline - time.time()))
            
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result
    
    if best_result is None:
        # Fallback: simple grid in 1x2 rectangle
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side * w
                    y = (j + 0.5) / side * h
                    r = min(w, h) / (2 * side) * 0.95
                    cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result


def _generate_initial(w, h, rng):
    """Generate initial circle positions and radii."""
    # Start with a grid arrangement
    cols = max(1, int(np.round(np.sqrt(N * w / h))))
    rows = max(1, int(np.ceil(N / cols)))
    
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(N):
        row = i // cols
        col = i % cols
        # Adjust for actual rows needed
        if row >= rows:
            row = rows - 1
            col = cols - 1
        
        x = (col + 0.5) / cols * w + rng.uniform(-0.02, 0.02)
        y = (row + 0.5) / rows * h + rng.uniform(-0.02, 0.02)
        
        centers[i] = [x, y]
        # Initial radius: fraction of cell size
        cell_w = w / cols
        cell_h = h / rows
        radii[i] = 0.4 * min(cell_w, cell_h)
    
    # Clip to stay inside rectangle
    for i in range(N):
        radii[i] = min(radii[i], centers[i][0], w - centers[i][0], 
                       centers[i][1], h - centers[i][1])
        radii[i] = max(radii[i], 0.01)
    
    return centers, radii


def _simulated_annealing(centers, radii, w, h, rng, time_budget):
    """Simulated annealing to optimize circle packing."""
    start_time = time.time()
    n = len(centers)
    
    # Current state
    cur_centers = centers.copy()
    cur_radii = radii.copy()
    cur_value = _evaluate(cur_centers, cur_radii, w, h)
    
    if cur_value is None:
        return None
    
    best_centers = cur_centers.copy()
    best_radii = cur_radii.copy()
    best_value = cur_value
    
    # Annealing schedule
    T = 0.1
    T_min = 1e-6
    alpha = 0.995  # cooling rate
    
    max_steps = 50000
    
    for step in range(max_steps):
        if time.time() > start_time + time_budget:
            break
        
        # Generate a random perturbation
        new_centers = cur_centers.copy()
        new_radii = cur_radii.copy()
        
        # Perturb 1-3 circles
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        for i in idx:
            # Perturb position
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            new_centers[i, 0] += dx
            new_centers[i, 1] += dy
            
            # Perturb radius
            dr = rng.normal(0, 0.01)
            new_radii[i] += dr
        
        # Clip to valid range
        for i in range(n):
            new_centers[i, 0] = np.clip(new_centers[i, 0], 0.01, w - 0.01)
            new_centers[i, 1] = np.clip(new_centers[i, 1], 0.01, h - 0.01)
            new_radii[i] = np.clip(new_radii[i], 0.01, min(w, h) / 2)
        
        # Evaluate new state
        new_value = _evaluate(new_centers, new_radii, w, h)
        
        if new_value is None:
            continue
        
        # Accept or reject
        delta = new_value - cur_value
        if delta > 0 or rng.random() < np.exp(delta / T):
            cur_centers = new_centers
            cur_radii = new_radii
            cur_value = new_value
            
            if cur_value > best_value:
                best_value = cur_value
                best_centers = cur_centers.copy()
                best_radii = cur_radii.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Return best result
    if best_value <= 0:
        return None
    
    result = np.column_stack([best_centers, best_radii])
    return result


def _evaluate(centers, radii, w, h):
    """Evaluate a configuration. Returns sum of radii if valid, None if invalid."""
    n = len(centers)
    
    # Check all circles are inside rectangle
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > w + 1e-9 or y - r < -1e-9 or y + r > h + 1e-9:
            return None
        if r < 1e-6:
            return None
    
    # Check no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return None
    
    return np.sum(radii)
