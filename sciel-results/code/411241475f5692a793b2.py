import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26

def _elapsed(t0):
    return time.time() - t0

def _remaining(t0, deadline):
    return max(0, deadline - time.time())

def _softplus(x):
    # Numerically stable softplus
    return np.where(x > 30, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))

def _validate(circles):
    """Check if circles are valid: inside unit square and non-overlapping."""
    if circles.shape != (N, 3):
        return False
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Inside square
    if np.any(x < -1e-9) or np.any(y < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    # Non-overlapping
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(r.max() * 2 + 1e-6)
    for i, j in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < r[i] + r[j] - 1e-7:
            return False
    return True

def _enforce_validity(circles):
    """Project circles to be inside the square with radius <= distance to boundary."""
    circles = circles.copy()
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Clip positions to [0,1]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    # Radius cannot exceed distance to nearest wall
    d_x = np.minimum(x, 1 - x)
    d_y = np.minimum(y, 1 - y)
    r = np.minimum(r, np.minimum(d_x, d_y))
    r = np.maximum(r, 0)
    circles[:, 0] = x
    circles[:, 1] = y
    circles[:, 2] = r
    return circles

def _objective(circles):
    """
    Objective: minimize -sum(radii) + penalty for overlaps and boundary violations.
    """
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Negative sum of radii
    obj = -np.sum(r)
    
    # Boundary penalty
    d_x = np.minimum(x, 1 - x)
    d_y = np.minimum(y, 1 - y)
    wall_viol = np.maximum(0, r - d_x) + np.maximum(0, r - d_y)
    obj += 100 * np.sum(wall_viol**2)
    
    # Overlap penalty
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(r.max() * 2 + 1e-6)
    overlap_pen = 0
    for i, j in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        viol = np.maximum(0, r[i] + r[j] - dist)
        overlap_pen += viol**2
    obj += 100 * overlap_pen
    
    return obj

def _objective_grad(circles):
    """
    Gradient of the objective function.
    """
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    grad = np.zeros_like(circles)
    
    # Base gradient: -1 for radii
    grad[:, 2] = -1.0
    
    # Boundary gradient
    d_x = np.minimum(x, 1 - x)
    d_y = np.minimum(y, 1 - y)
    
    # For x: if x < 0.5, d_x = x, else d_x = 1-x
    # wall_viol_x = max(0, r - d_x)
    # d(wall_viol_x)/dx = -1 if x < 0.5 and r > d_x, else +1 if x > 0.5 and r > d_x
    for i in range(N):
        if r[i] > d_x[i]:
            if x[i] < 0.5:
                grad[i, 0] += 200 * (r[i] - d_x[i]) * (-1)
            else:
                grad[i, 0] += 200 * (r[i] - d_x[i]) * (1)
        if r[i] > d_y[i]:
            if y[i] < 0.5:
                grad[i, 1] += 200 * (r[i] - d_y[i]) * (-1)
            else:
                grad[i, 1] += 200 * (r[i] - d_y[i]) * (1)
    
    # Overlap gradient
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(r.max() * 2 + 1e-6)
    for i, j in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        viol = r[i] + r[j] - dist
        if viol > 0:
            # d(viol^2)/dx_i = 2*viol * d(viol)/dx_i
            # d(viol)/dx_i = -dx_i / dist
            inv_dist = 1.0 / dist if dist > 1e-12 else 0.0
            grad[i, 0] += 200 * viol * (-dx * inv_dist)
            grad[j, 0] += 200 * viol * (dx * inv_dist)
            grad[i, 1] += 200 * viol * (-dy * inv_dist)
            grad[j, 1] += 200 * viol * (dy * inv_dist)
            grad[i, 2] += 200 * viol * (1)
            grad[j, 2] += 200 * viol * (1)
    
    return grad

def _initial_grid():
    """Initialize circles on a grid."""
    circles = np.zeros((N, 3))
    # 6x5 grid = 30, take 26
    side_x, side_y = 6, 5
    count = 0
    for i in range(side_x):
        for j in range(side_y):
            if count >= N:
                break
            circles[count, 0] = (i + 0.5) / side_x
            circles[count, 1] = (j + 0.5) / side_y
            circles[count, 2] = 0.5 / max(side_x, side_y) - 0.01
            count += 1
        if count >= N:
            break
    return _enforce_validity(circles)

def _initial_hex():
    """Initialize circles in a hexagonal pattern."""
    circles = np.zeros((N, 3))
    # Hexagonal packing
    spacing = 0.18
    count = 0
    for row in range(10):
        y = 0.1 + row * spacing * np.sqrt(3) / 2
        if y > 0.9:
            break
        offset = 0 if row % 2 == 0 else spacing / 2
        for col in range(10):
            x = 0.1 + col * spacing + offset
            if x > 0.9:
                break
            if count >= N:
                break
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = spacing / 2 - 0.005
            count += 1
        if count >= N:
            break
    # If we didn't fill all N, add random
    if count < N:
        for k in range(count, N):
            circles[k, 0] = np.random.uniform(0.1, 0.9)
            circles[k, 1] = np.random.uniform(0.1, 0.9)
            circles[k, 2] = 0.05
    return _enforce_validity(circles)

def _polish_analytic(circles, deadline):
    """
    Use scipy.optimize.minimize with L-BFGS-B to polish the configuration.
    """
    if _elapsed(0) > 0:  # dummy to avoid unused warning
        pass
    
    x0 = circles.flatten()
    
    # Bounds: x, y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    start_time = time.time()
    
    def obj_flat(x):
        c = x.reshape(N, 3)
        return _objective(c)
    
    def grad_flat(x):
        c = x.reshape(N, 3)
        return _objective_grad(c).flatten()
    
    result = minimize(
        obj_flat, x0,
        jac=grad_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    best = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    if result.success or result.fun < obj_flat(x0):
        new_circles = result.x.reshape(N, 3)
        new_circles = _enforce_validity(new_circles)
        if _validate(new_circles):
            score = np.sum(new_circles[:, 2])
            if score > best_score:
                best = new_circles
                best_score = score
    
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin
    
    np.random.seed(42)
    
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initializations
    initializations = [
        _initial_grid,
        _initial_hex
    ]
    
    # Add some random initializations
    for _ in range(3):
        def random_init():
            circles = np.zeros((N, 3))
            for i in range(N):
                circles[i, 0] = np.random.uniform(0.1, 0.9)
                circles[i, 1] = np.random.uniform(0.1, 0.9)
                circles[i, 2] = np.random.uniform(0.03, 0.08)
            return _enforce_validity(circles)
        initializations.append(random_init)
    
    for init_func in initializations:
        if time.time() > deadline - 15:
            break
        
        circles = init_func()
        
        # Polish with scipy
        polished = _polish_analytic(circles, deadline)
        
        if _validate(polished):
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_circles = polished.copy()
        
        # Also try the unpolished if it's valid and better
        if _validate(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    if best_circles is None:
        # Fallback: return a valid grid configuration
        best_circles = _initial_grid()
    
    return best_circles
