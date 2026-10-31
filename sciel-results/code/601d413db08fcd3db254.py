"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 26
DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _soft_plus(u, beta=200.0):
    """Smooth approximation of max(0, u) using log-sum-exp."""
    # Numerically stable
    if u > 50.0 / beta:
        return u
    if u < -50.0 / beta:
        return 0.0
    return np.logaddexp(0.0, beta * u) / beta

def _soft_plus_grad(u, beta=200.0):
    """Derivative of soft_plus wrt u."""
    # sigmoid(beta * u)
    if beta * u > 50:
        return 1.0
    if beta * u < -50:
        return 0.0
    z = np.exp(-beta * u)
    return 1.0 / (1.0 + z)

def _check_feasibility(circles, tol=1e-9):
    """Check if circles are all within [0,1]^2 and non-overlapping."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol or x - r < -tol or y - r < -tol or x + r > 1.0 + tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = math.hypot(x1-x2, y1-y2)
            if dist < (r1 + r2) - tol:
                return False
    return True

def _make_feasible(circles, shrink_iter=20):
    """If circles overlap or go out of bounds, shrink radii slightly to make feasible."""
    circles = circles.copy()
    for _ in range(shrink_iter):
        feasible = _check_feasibility(circles, tol=1e-9)
        if feasible:
            return circles
        # Shrink all radii by 1%
        circles[:, 2] *= 0.99
        # Also pull centers back into bounds
        for i in range(len(circles)):
            r = circles[i, 2]
            circles[i, 0] = min(max(circles[i, 0], r), 1.0 - r)
            circles[i, 1] = min(max(circles[i, 1], r), 1.0 - r)
    return circles

def _init_grid_5x6():
    """Initialize 26 circles in a 5x6 grid pattern."""
    rows, cols = 5, 6
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            pts.append((x, y))
    # Take first 26
    pts = pts[:N]
    # Compute initial radii as half the minimum spacing
    min_r = 0.5 / cols - 1e-4
    circles = []
    for x, y in pts:
        circles.append([x, y, min_r])
    return np.array(circles)

def _init_random(seed=0):
    """Random initialization inside the unit square."""
    rng = np.random.RandomState(seed)
    circles = []
    for _ in range(N):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.03
        circles.append([x, y, r])
    return np.array(circles)

def _objective(z, beta=200.0):
    """
    Objective: minimize -sum(r_i) + lambda * (pairwise overlap penalties + boundary penalties)
    z = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    obj = -np.sum(r)
    
    lam = 50.0
    
    # Pairwise overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx*dx + dy*dy
            # Want dist >= r_i + r_j
            # Penalty: soft_plus((r_i + r_j)^2 - dist2)
            # Actually, use soft_plus(r_i + r_j - dist) is more geometrically meaningful
            # But dist = sqrt(dist2), non-differentiable at 0. Use dist2 formulation.
            # Let u = (r_i + r_j)^2 - dist2. If u > 0, circles overlap.
            u = (r[i] + r[j])**2 - dist2
            obj += lam * _soft_plus(u, beta)
    
    # Boundary penalties
    for i in range(n):
        # Left: x_i - r_i >= 0 => penalty soft_plus(r_i - x_i)
        obj += lam * _soft_plus(r[i] - x[i], beta)
        # Right: 1 - x_i - r_i >= 0 => penalty soft_plus(r_i + x_i - 1)
        obj += lam * _soft_plus(r[i] + x[i] - 1.0, beta)
        # Bottom: y_i - r_i >= 0
        obj += lam * _soft_plus(r[i] - y[i], beta)
        # Top: 1 - y_i - r_i >= 0
        obj += lam * _soft_plus(r[i] + y[i] - 1.0, beta)
    
    return obj

def _gradient(z, beta=200.0):
    """
    Gradient of the objective.
    """
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    
    grad = np.zeros(3 * n)
    lam = 50.0
    
    # Objective gradient from -sum(r): -1 for each r component
    grad[2::3] = -1.0
    
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx*dx + dy*dy
            if dist2 < 1e-20:
                dist2 = 1e-20
            dist = math.sqrt(dist2)
            
            u = (r[i] + r[j])**2 - dist2
            gu = _soft_plus_grad(u, beta)
            if gu == 0.0:
                continue
            
            # d(u)/d(x_i) = -2*dx = 2*(x_j - x_i)
            # d(u)/d(x_j) = 2*dx
            # d(u)/d(y_i) = -2*dy
            # d(u)/d(y_j) = 2*dy
            # d(u)/d(r_i) = 2*(r_i + r_j)
            # d(u)/d(r_j) = 2*(r_i + r_j)
            
            factor = 2.0 / dist
            grad[3*i + 0] += lam * gu * (-2.0 * dx)
            grad[3*i + 1] += lam * gu * (-2.0 * dy)
            grad[3*j + 0] += lam * gu * (2.0 * dx)
            grad[3*j + 1] += lam * gu * (2.0 * dy)
            grad[3*i + 2] += lam * gu * 2.0 * (r[i] + r[j])
            grad[3*j + 2] += lam * gu * 2.0 * (r[i] + r[j])
    
    # Boundary
    for i in range(n):
        # Left
        uL = r[i] - x[i]
        guL = _soft_plus_grad(uL, beta)
        grad[3*i + 0] += lam * guL * (-1.0)
        grad[3*i + 2] += lam * guL * (1.0)
        
        # Right
        uR = r[i] + x[i] - 1.0
        guR = _soft_plus_grad(uR, beta)
        grad[3*i + 0] += lam * guR * (1.0)
        grad[3*i + 2] += lam * guR * (1.0)
        
        # Bottom
        uB = r[i] - y[i]
        guB = _soft_plus_grad(uB, beta)
        grad[3*i + 1] += lam * guB * (-1.0)
        grad[3*i + 2] += lam * guB * (1.0)
        
        # Top
        uT = r[i] + y[i] - 1.0
        guT = _soft_plus_grad(uT, beta)
        grad[3*i + 1] += lam * guT * (1.0)
        grad[3*i + 2] += lam * guT * (1.0)
    
    return grad

def construct():
    start_time = time.time()
    
    # Bounds
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    best_circles = None
    best_score = -1.0
    
    # Try multiple initializations
    inits = []
    # Grid 5x6
    inits.append(_init_grid_5x6())
    # Random inits
    for seed in [1, 2, 3, 4, 5]:
        inits.append(_init_random(seed))
    
    for init_idx, init_circles in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
        
        z0 = init_circles.reshape(-1)
        
        # Limit time per run
        time_per_run = min(remaining / max(1, len(inits) - init_idx), 15.0)
        
        try:
            res = minimize(
                _objective,
                z0,
                jac=_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            circles = res.x.reshape(N, 3)
            # Clip to bounds
            circles[:, 0] = np.clip(circles[:, 0], 0, 1)
            circles[:, 1] = np.clip(circles[:, 1], 0, 1)
            circles[:, 2] = np.clip(circles[:, 2], 1e-6, 0.5)
            
            # Make feasible
            circles = _make_feasible(circles)
            
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                
        except Exception:
            pass
    
    # If no valid result, fall back to grid
    if best_circles is None or not _check_feasibility(best_circles):
        best_circles = _init_grid_5x6()
        best_circles = _make_feasible(best_circles)
    
    # Final check
    if not _check_feasibility(best_circles):
        # Force feasibility by shrinking
        for _ in range(50):
            if _check_feasibility(best_circles):
                break
            best_circles[:, 2] *= 0.95
    
    return best_circles

# EVOLVE-BLOCK-END
