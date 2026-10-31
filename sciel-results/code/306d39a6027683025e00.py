"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_validity(circles, w, h):
    """Check if circles are valid: inside rectangle, non-overlapping."""
    n = len(circles)
    for i in range(n):
        xi, yi, ri = circles[i]
        if ri <= 0:
            return False
        if xi - ri < -1e-9 or xi + ri > w + 1e-9:
            return False
        if yi - ri < -1e-9 or yi + ri > h + 1e-9:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _score(circles, w, h):
    """Compute the objective: sum of radii if valid, else large penalty."""
    if not _check_validity(circles, w, h):
        return -1e6
    return np.sum(circles[:, 2])

def _objective(params, w, h, n):
    """Objective for optimizer: negative sum of radii (we minimize)."""
    circles = params.reshape(n, 3)
    # Add penalty for constraint violations
    penalty = 0.0
    for i in range(n):
        xi, yi, ri = circles[i]
        if ri <= 0:
            penalty += 1e4
        if xi - ri < 0:
            penalty += 1e4 * (0 - (xi - ri))**2
        if xi + ri > w:
            penalty += 1e4 * ((xi + ri) - w)**2
        if yi - ri < 0:
            penalty += 1e4 * (0 - (yi - ri))**2
        if yi + ri > h:
            penalty += 1e4 * ((yi + ri) - h)**2
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj:
                penalty += 1e4 * (ri + rj - dist)**2
    return -np.sum(circles[:, 2]) + penalty

def _solve_for_aspect(w, h, n, max_iter=5000):
    """Solve the packing problem for given aspect ratio."""
    # Initialize circles in a grid pattern
    side = int(np.ceil(np.sqrt(n * w / h)))
    if side < 2:
        side = 2
    rows = int(np.ceil(n / side))
    
    circles = np.zeros((n, 3))
    for idx in range(n):
        i = idx % side
        j = idx // side
        x = (i + 0.5) * w / side
        y = (j + 0.5) * h / rows
        r = min(w / side, h / rows) * 0.45
        circles[idx] = [x, y, r]
    
    best_params = circles.copy()
    best_score = _score(circles, w, h)
    
    # Use Nelder-Mead for optimization
    x0 = circles.flatten()
    
    result = minimize(
        _objective, x0, args=(w, h, n),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    if result.fun < 1e5:  # Check if we got a reasonable result
        new_circles = result.x.reshape(n, 3)
        new_score = _score(new_circles, w, h)
        if new_score > best_score:
            best_params = new_circles
            best_score = new_score
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = N
    best_circles = None
    best_score = 0.0
    best_w = None
    best_h = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 0.5, 1.2, 0.8, 1.8, 2.5, 0.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        
        circles, score = _solve_for_aspect(w, h, n, max_iter=3000)
        
        if score > best_score:
            best_score = score
            best_circles = circles
            best_w = w
            best_h = h
    
    if best_circles is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:n])
    
    return best_circles
# EVOLVE-BLOCK-END
