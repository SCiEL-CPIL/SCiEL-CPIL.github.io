"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140  # seconds, leave margin before 150s kill

def _violations(circles):
    """Compute constraint violations for a set of circles."""
    n = len(circles)
    violations = 0.0
    # Boundary violations
    for i in range(n):
        x, y, r = circles[i]
        if x < r:
            violations += (r - x) ** 2
        if x > 1 - r:
            violations += (x - (1 - r)) ** 2
        if y < r:
            violations += (r - y) ** 2
        if y > 1 - r:
            violations += (y - (1 - r)) ** 2
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < min_dist ** 2:
                violations += (min_dist - np.sqrt(dist_sq)) ** 2
    return violations

def _objective_neg(circles):
    """Negative sum of radii (we minimize negative to maximize sum)."""
    return -np.sum(circles[:, 2])

def _optimize_from_init(init_circles, max_iter=500, tol=1e-8):
    """Optimize from initial configuration using penalty method."""
    x0 = init_circles.flatten()
    
    def penalty_objective(x):
        circles = x.reshape(-1, 3)
        # Ensure radii are positive
        circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
        obj = -np.sum(circles[:, 2])
        pen = _violations(circles)
        return obj + 1000.0 * pen
    
    result = minimize(penalty_objective, x0, method='BFGS', 
                      options={'maxiter': max_iter, 'gtol': tol})
    
    if result.success or result.fun < penalty_objective(x0):
        best_circles = result.x.reshape(-1, 3)
        best_circles[:, 2] = np.maximum(best_circles[:, 2], 1e-6)
        return best_circles, _objective_neg(best_circles)
    else:
        return init_circles, _objective_neg(init_circles)

def _grid_init(n, spacing_factor=1.0):
    """Grid-based initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * spacing_factor
            cells.append([x, y, r])
    return np.array(cells[:n])

def _random_init(n, rng):
    """Random initialization with small circles."""
    circles = []
    for _ in range(n):
        r = rng.uniform(0.01, 0.05)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)

def _hex_init(n, rng):
    """Hexagonal packing initialization."""
    circles = []
    row = 0
    col = 0
    while len(circles) < n:
        if row % 2 == 0:
            x = 0.5 / (2 * col + 1) + col * (1.0 / (2 * col + 1))
        else:
            x = 0.5 / (2 * col + 2) + col * (1.0 / (2 * col + 2))
        y = (row + 0.5) / (n + 1)
        r = 0.4 / (2 * col + 1)
        if x < 1 - r and y < 1 - r:
            circles.append([x, y, r])
        col += 1
        if col > 2 * n:
            col = 0
            row += 1
    return np.array(circles[:n])

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_circles = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    inits = []
    
    # Grid-based
    for sf in [0.8, 0.9, 1.0, 1.1]:
        inits.append(_grid_init(N, sf))
    
    # Random
    for _ in range(5):
        inits.append(_random_init(N, rng))
    
    # Hexagonal
    inits.append(_hex_init(N, rng))
    
    # More random with different radii ranges
    for _ in range(5):
        circles = []
        for _ in range(N):
            r = rng.uniform(0.02, 0.08)
            x = rng.uniform(r, 1 - r)
            y = rng.uniform(r, 1 - r)
            circles.append([x, y, r])
        inits.append(np.array(circles))
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        circles, score = _optimize_from_init(init, max_iter=300)
        
        if score > best_score:
            best_score = score
            best_circles = circles
    
    # Final polish on best solution
    if best_circles is not None and time.time() - start_time < DEADLINE - 5:
        circles, score = _optimize_from_init(best_circles, max_iter=500, tol=1e-10)
        if score > best_score:
            best_circles = circles
            best_score = score
    
    # Ensure validity
    if best_circles is None:
        # Fallback to grid
        best_circles = _grid_init(N, 0.9)
    
    # Clip to ensure within bounds and positive radii
    best_circles = best_circles.copy()
    best_circles[:, 2] = np.maximum(best_circles[:, 2], 1e-6)
    best_circles[:, 0] = np.clip(best_circles[:, 0], best_circles[:, 2], 1 - best_circles[:, 2])
    best_circles[:, 1] = np.clip(best_circles[:, 1], best_circles[:, 2], 1 - best_circles[:, 2])
    
    return best_circles
# EVOLVE-BLOCK-END
