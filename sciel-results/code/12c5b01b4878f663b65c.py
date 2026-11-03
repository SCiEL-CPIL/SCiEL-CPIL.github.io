import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_radii(pos, w, h):
    """Compute maximum radii for given positions in w x h rectangle."""
    n = len(pos)
    radii = np.zeros(n)
    for i in range(n):
        min_dist = min(
            pos[i, 0],
            w - pos[i, 0],
            pos[i, 1],
            h - pos[i, 1]
        )
        for j in range(n):
            if i != j:
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                           (pos[i, 1] - pos[j, 1])**2)
                min_dist = min(min_dist, d / 2.0)
        radii[i] = max(0, min_dist - 1e-9)
    return radii

def _pack_for_aspect(w, h, n_circles=N, time_budget=5.0):
    """Pack circles in w x h rectangle, return (positions, radii) maximizing sum of radii."""
    start = time.time()
    
    # Initialize in a staggered grid for better initial packing
    cols = int(np.ceil(np.sqrt(n_circles * w / h)))
    rows = int(np.ceil(n_circles / cols))
    
    positions = np.zeros((n_circles, 2))
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        # Stagger odd rows
        offset = 0.5 * w / cols if row % 2 == 1 else 0
        positions[i, 0] = (col + 0.5) * w / cols + offset
        positions[i, 1] = (row + 0.5) * h / rows
        # Clamp to bounds
        positions[i, 0] = np.clip(positions[i, 0], 0, w)
        positions[i, 1] = np.clip(positions[i, 1], 0, h)
    
    # Objective function for optimization
    def objective(x):
        pos = x[:n_circles * 2].reshape(n_circles, 2)
        # Check bounds - return large penalty if out of bounds
        if np.any(pos[:, 0] < -0.01) or np.any(pos[:, 0] > w + 0.01) or \
           np.any(pos[:, 1] < -0.01) or np.any(pos[:, 1] > h + 0.01):
            return 1e6
        # Clamp for radius calculation
        pos_clamped = np.clip(pos, 0, [w, h])
        radii = _compute_radii(pos_clamped, w, h)
        return -np.sum(radii)
    
    # Try multiple optimization runs with different initializations
    best_pos = positions.copy()
    best_score = -np.sum(_compute_radii(positions, w, h))
    
    # Run 1: Nelder-Mead from staggered grid
    x0 = positions.flatten()
    bounds = [(0, w)] * n_circles + [(0, h)] * n_circles
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         bounds=bounds,
                         options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        pos = result.x[:n_circles * 2].reshape(n_circles, 2)
        pos = np.clip(pos, 0, [w, h])
        score = -np.sum(_compute_radii(pos, w, h))
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return best_pos, _compute_radii(best_pos, w, h)
    
    # Run 2: COBYLA from different initialization
    # Try a more uniform random initialization
    np.random.seed(42)
    pos_rand = np.column_stack([np.random.uniform(0, w, n_circles), 
                                np.random.uniform(0, h, n_circles)])
    x0_rand = pos_rand.flatten()
    
    try:
        result = minimize(objective, x0_rand, method='COBYLA',
                         options={'maxiter': 2000, 'rhobeg': 0.1, 'catol': 1e-8})
        pos = result.x[:n_circles * 2].reshape(n_circles, 2)
        pos = np.clip(pos, 0, [w, h])
        score = -np.sum(_compute_radii(pos, w, h))
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return best_pos, _compute_radii(best_pos, w, h)
    
    # Run 3: Polish the best result with L-BFGS-B
    try:
        result = minimize(objective, best_pos.flatten(), method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        pos = result.x[:n_circles * 2].reshape(n_circles, 2)
        pos = np.clip(pos, 0, [w, h])
        score = -np.sum(_compute_radii(pos, w, h))
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
    except Exception:
        pass
    
    radii = _compute_radii(best_pos, w, h)
    return best_pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    best_score = 0
    best_result = None
    
    # Try various aspect ratios - focus on promising ranges
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    # Try aspect ratios from 0.4 to 2.5 (w/h)
    aspect_ratios = np.concatenate([
        np.linspace(0.4, 1.0, 8),
        np.linspace(1.0, 2.5, 12)
    ])
    
    # Sort by how close to square (squares often pack well)
    aspect_ratios = sorted(aspect_ratios, key=lambda ar: abs(ar - 1.0))
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Time budget per aspect ratio
        remaining = deadline - 5 - time.time()
        n_remaining = len(aspect_ratios) - list(aspect_ratios).index(ar)
        time_budget = max(2.0, min(8.0, remaining / max(1, n_remaining)))
        
        pos, radii = _pack_for_aspect(w, h, time_budget=time_budget)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
    
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
