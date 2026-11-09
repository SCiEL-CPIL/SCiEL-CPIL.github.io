# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # No overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))

def _penalty(centers, sides, tol=1e-7):
    """Penalty for constraint violations."""
    n = len(sides)
    penalty = 0.0
    # Inside unit square
    penalty += np.sum(np.maximum(0, -tol - (centers[:, 0] - sides / 2))**2)
    penalty += np.sum(np.maximum(0, -tol - (centers[:, 1] - sides / 2))**2)
    penalty += np.sum(np.maximum(0, (centers[:, 0] + sides / 2) - (1 + tol))**2)
    penalty += np.sum(np.maximum(0, (centers[:, 1] + sides / 2) - (1 + tol))**2)
    # Positive sides
    penalty += np.sum(np.maximum(0, -sides)**2)
    # No overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # If dx < t and dy < t, they overlap.
            # Penalty is the amount of "penetration" into the forbidden zone.
            # Forbidden: dx < t AND dy < t.
            # We want to push them apart so that dx >= t OR dy >= t.
            # A simple penalty: if both are < t, add the product of violations?
            # Or just add (t - dx)^2 + (t - dy)^2 if both are < t?
            # Better: if dx < t and dy < t, penalty += (t - dx)**2 + (t - dy)**2
            if dx < t and dy < t:
                penalty += (t - dx)**2 + (t - dy)**2
    return penalty

def _total_objective(params):
    """Objective for minimizer: minimize -sum(sides) + penalty."""
    n = 10
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    # Ensure sides are positive in objective calculation to avoid weirdness
    sides_safe = np.maximum(sides, 1e-6)
    
    obj = -np.sum(sides_safe) + 100.0 * _penalty(centers, sides_safe)
    return obj

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    best_centers = None
    best_sides = None
    best_val = 0.0

    # Define several initial guesses
    inits = []
    
    # 1. 3x3 grid (one empty)
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    # Add 10th square in the empty spot (0,0) but smaller?
    # Or just place it randomly?
    # Let's place it in the center of the empty cell (0,0) with size 0.1
    centers_init = np.zeros((10, 2))
    sides_init = np.zeros(10)
    centers_init[:9] = np.stack([cx, cy], axis=1)
    sides_init[:9] = side
    centers_init[9] = [0.5 * side, 0.5 * side]
    sides_init[9] = 0.1
    inits.append(np.concatenate([centers_init.flatten(), sides_init]))

    # 2. 4x4 grid (16 squares, take 10)
    k = 4
    side = 1.0 / k
    idx = np.arange(10)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers_init = np.stack([cx, cy], axis=1)
    sides_init = np.full(10, side)
    inits.append(np.concatenate([centers_init.flatten(), sides_init]))

    # 3. Random uniform
    rng = np.random.default_rng(123)
    for _ in range(5):
        cx = rng.uniform(0.1, 0.9, 10)
        cy = rng.uniform(0.1, 0.9, 10)
        s = rng.uniform(0.1, 0.4, 10)
        centers_init = np.stack([cx, cy], axis=1)
        sides_init = s
        inits.append(np.concatenate([centers_init.flatten(), sides_init]))

    # 4. Structured: 4 corners, 1 center, 5 others
    a = 0.25
    b = 0.25
    h = 0.5 - a - b/2
    centers_init = np.zeros((10, 2))
    sides_init = np.zeros(10)
    
    # Corners
    centers_init[0] = [a/2, a/2]; sides_init[0] = a
    centers_init[1] = [1-a/2, a/2]; sides_init[1] = a
    centers_init[2] = [a/2, 1-a/2]; sides_init[2] = a
    centers_init[3] = [1-a/2, 1-a/2]; sides_init[3] = a
    
    # Center
    centers_init[4] = [0.5, 0.5]; sides_init[4] = b
    
    # Top 2
    centers_init[5] = [0.25, 1 - h/2]; sides_init[5] = h
    centers_init[6] = [0.75, 1 - h/2]; sides_init[6] = h
    
    # Bottom 2
    centers_init[7] = [0.25, h/2]; sides_init[7] = h
    centers_init[8] = [0.75, h/2]; sides_init[8] = h
    
    # Left 1
    centers_init[9] = [h/2, 0.5]; sides_init[9] = h
    
    inits.append(np.concatenate([centers_init.flatten(), sides_init]))

    # Optimize each initial guess
    for init_params in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(_total_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            params = res.x
            centers = params[:2*n].reshape(n, 2)
            sides = params[2*n:]
            
            # Clean up: clamp to valid range
            sides = np.maximum(sides, 1e-6)
            centers[:, 0] = np.clip(centers[:, 0], sides/2, 1 - sides/2)
            centers[:, 1] = np.clip(centers[:, 1], sides/2, 1 - sides/2)
            
            if _is_valid(centers, sides, tol):
                val = _objective(centers, sides)
                if val > best_val:
                    best_val = val
                    best_centers = centers.copy()
                    best_sides = sides.copy()
        except Exception:
            continue

    # Fallback to grid if optimization failed
    if best_centers is None:
        k = 3
        side = 1.0 / k
        idx = np.arange(9)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        centers[:9] = np.stack([cx, cy], axis=1)
        sides[:9] = side
        centers[9] = [0.5 * side, 0.5 * side]
        sides[9] = 0.1
        best_centers = centers
        best_sides = sides
        best_val = _objective(centers, sides)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
