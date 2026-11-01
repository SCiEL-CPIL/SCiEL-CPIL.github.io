"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist


def _check_valid(circles, eps=1e-7):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -eps or x + r > 1 + eps:
            return False
        if y - r < -eps or y + r > 1 + eps:
            return False
        if r < -eps:
            return False
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - eps:
                return False
    return True


def _penalty(circles):
    """Compute penalty for invalid circles. 0 if valid."""
    n = len(circles)
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        # Boundary violations
        if x - r < 0:
            penalty += (x - r)**2
        if x + r > 1:
            penalty += (x + r - 1)**2
        if y - r < 0:
            penalty += (y - r)**2
        if y + r > 1:
            penalty += (y + r - 1)**2
        if r < 0:
            penalty += r**2
    # Pairwise overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                penalty += overlap**2
    return penalty


def _objective(params):
    """Objective to minimize: negative sum of radii + penalty."""
    n = N
    circles = params.reshape(n, 3)
    neg_sum_r = -np.sum(circles[:, 2])
    pen = _penalty(circles)
    return neg_sum_r + 100.0 * pen


def _initial_guess():
    """Generate initial guess: grid placement with small radii."""
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
    # Pad if needed
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.01])
    return np.array(cells[:N]).flatten()


def construct():
    """Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii."""
    x0 = _initial_guess()
    
    # Define bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    # Try multiple random restarts to find good solution
    best_circles = None
    best_value = -np.inf
    
    for attempt in range(5):
        if attempt == 0:
            x0_try = _initial_guess()
        else:
            # Perturb initial guess
            x0_try = _initial_guess() + np.random.randn(N * 3) * 0.05
            # Clip to valid ranges
            for i in range(N):
                x0_try[i * 3] = np.clip(x0_try[i * 3], 0, 1)
                x0_try[i * 3 + 1] = np.clip(x0_try[i * 3 + 1], 0, 1)
                x0_try[i * 3 + 2] = np.clip(x0_try[i * 3 + 2], 0.01, 0.4)
        
        result = minimize(
            _objective,
            x0_try,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        circles = result.x.reshape(N, 3)
        # Ensure radii are non-negative
        circles[:, 2] = np.maximum(circles[:, 2], 0)
        
        # Check validity
        if _check_valid(circles):
            value = np.sum(circles[:, 2])
            if value > best_value:
                best_value = value
                best_circles = circles.copy()
    
    if best_circles is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
