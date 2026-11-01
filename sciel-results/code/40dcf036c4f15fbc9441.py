"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def _validate(arr):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r < 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-7:
                return False
    return True

def _objective(pos_r, n):
    """Objective function: minimize negative sum of radii + penalty for violations."""
    pos = pos_r[:2*n]
    radii = pos_r[2*n:]
    
    # Penalty for going outside [0,1]^2
    penalty = 0.0
    for i in range(n):
        x, y, r = pos[2*i], pos[2*i+1], radii[i]
        if r < 0:
            penalty += 1000 * (-r)
        if x - r < 0:
            penalty += 1000 * (r - x)
        if x + r > 1:
            penalty += 1000 * (x + r - 1)
        if y - r < 0:
            penalty += 1000 * (r - y)
        if y + r > 1:
            penalty += 1000 * (y + r - 1)
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = pos[2*i], pos[2*i+1], radii[i]
            xj, yj, rj = pos[2*j], pos[2*j+1], radii[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = ri + rj - dist
            if overlap > 0:
                penalty += 100 * overlap**2
    
    return -np.sum(radii) + penalty

def _initial_grid():
    """Create an initial grid arrangement."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
    return np.array(cells[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with grid
    best_arr = _initial_grid()
    best_score = np.sum(best_arr[:, 2])
    
    if not _validate(best_arr):
        # Fallback: very small circles
        best_arr = np.array([[0.5, 0.5, 0.01]] * N)
        best_score = np.sum(best_arr[:, 2])
    
    # Try multiple random restarts with local optimization
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Create initial configuration: grid with some perturbation
        init = _initial_grid().copy()
        
        # Add random perturbation
        for i in range(N):
            init[i, 0] += random.uniform(-0.02, 0.02)
            init[i, 1] += random.uniform(-0.02, 0.02)
            init[i, 2] = max(0.001, init[i, 2] * random.uniform(0.8, 1.2))
        
        # Flatten to single vector for optimization
        x0 = np.concatenate([init[:, :2].flatten(), init[:, 2]])
        
        # Bounds: positions in [0,1], radii in [0, 0.5]
        bounds = []
        for i in range(2*N):
            bounds.append((0, 1))
        for i in range(N):
            bounds.append((0.001, 0.5))
        
        # Optimize
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success or res.fun < 1e-4:
                res_pos = res.x[:2*N].reshape(N, 2)
                res_radii = res.x[2*N:]
                res_arr = np.column_stack([res_pos, res_radii])
                
                if _validate(res_arr):
                    score = np.sum(res_radii)
                    if score > best_score:
                        best_score = score
                        best_arr = res_arr
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final validation
    if not _validate(best_arr):
        # Return a valid fallback
        side = 6
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        best_arr = np.array(cells[:N])
    
    return best_arr
# EVOLVE-BLOCK-END
