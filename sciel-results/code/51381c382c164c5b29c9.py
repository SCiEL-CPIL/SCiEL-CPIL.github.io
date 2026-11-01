"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.spatial.distance import cdist

N = 21
DEADLINE = 110.0  # seconds, safety margin before 120s limit
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _validate(circles, tol=1e-6):
    """Check if circles are valid: non-negative radii, disjoint interiors, bounding rect perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    if np.any(circles[:, 2] < -tol):
        return False
    # Bounding box
    x_min = np.min(circles[:, 0] - circles[:, 2])
    x_max = np.max(circles[:, 0] + circles[:, 2])
    y_min = np.min(circles[:, 1] - circles[:, 2])
    y_max = np.max(circles[:, 1] + circles[:, 2])
    w = x_max - x_min
    h = y_max - y_min
    if 2 * (w + h) > 4.0 + 1e-4:
        return False
    # Disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((circles[i, 0] - circles[j, 0]) ** 2 + (circles[i, 1] - circles[j, 1]) ** 2)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True


def _objective(circles):
    """Return sum of radii (we maximize, so minimize negative)."""
    return -np.sum(circles[:, 2])


def _init_grid():
    """Initialize circles in a grid pattern within a 2x2 square (perimeter 8... wait, perimeter 4 means w+h=2).
    
    Perimeter 4: 2(w+h) = 4 => w+h = 2.
    Let's use a 1x1 square: perimeter = 2(1+1) = 4. Good.
    Or a 2x0 rectangle? No, circles need space.
    Let's use a square of side 1: perimeter = 4.
    """
    side = int(np.ceil(np.sqrt(N)))
    # Place in a 1x1 square
    xs, ys = np.meshgrid(np.linspace(0.5/side, 1 - 0.5/side, side),
                         np.linspace(0.5/side, 1 - 0.5/side, side))
    xs = xs.flatten()[:N]
    ys = ys.flatten()[:N]
    r = 0.5 / side - 1e-3
    circles = np.column_stack([xs, ys, np.full(N, r)])
    return circles


def _init_random():
    """Random initialization in a 1x1 square."""
    xs = np.random.uniform(0.1, 0.9, N)
    ys = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    return np.column_stack([xs, ys, r])


def _scale_to_perimeter(circles):
    """Scale the configuration so that the bounding rectangle has perimeter exactly 4."""
    x_min = np.min(circles[:, 0] - circles[:, 2])
    x_max = np.max(circles[:, 0] + circles[:, 2])
    y_min = np.min(circles[:, 1] - circles[:, 2])
    y_max = np.max(circles[:, 1] + circles[:, 2])
    w = x_max - x_min
    h = y_max - y_min
    if w + h <= 0:
        return circles
    # We want 2(w+h) = 4 => w+h = 2
    # Current perimeter = 2(w+h)
    # Scale factor s such that s*w + s*h = 2 => s = 2/(w+h)
    s = 2.0 / (w + h)
    # Scale positions and radii relative to the center of the bounding box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    new_circles = circles.copy()
    new_circles[:, 0] = cx + s * (circles[:, 0] - cx)
    new_circles[:, 1] = cy + s * (circles[:, 1] - cy)
    new_circles[:, 2] = s * circles[:, 2]
    return new_circles


def _penalty_objective(circles, penalty_weight=100.0):
    """Penalty-based objective for unconstrained optimization."""
    x = circles
    # Objective: maximize sum of radii
    obj = -np.sum(x[:, 2])
    
    # Penalty for overlapping circles
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2)
            overlap = x[i, 2] + x[j, 2] - dist
            if overlap > 0:
                penalty += overlap**2
    
    # Penalty for negative radii
    neg_r = np.clip(-x[:, 2], 0, None)
    penalty += np.sum(neg_r**2)
    
    # Penalty for perimeter > 4
    x_min = np.min(x[:, 0] - x[:, 2])
    x_max = np.max(x[:, 0] + x[:, 2])
    y_min = np.min(x[:, 1] - x[:, 2])
    y_max = np.max(x[:, 1] + x[:, 2])
    w = x_max - x_min
    h = y_max - y_min
    perim = 2 * (w + h)
    if perim > 4:
        penalty += (perim - 4)**2
    
    return obj + penalty_weight * penalty


def _solve_constrained(circles0, max_iter=1000, tol=1e-8):
    """Solve using scipy.optimize.minimize with SLSQP and constraints."""
    n = N
    # Variables: x_i, y_i, r_i for each circle
    # Constraints:
    # 1. r_i >= 0 for all i
    # 2. dist(i,j) >= r_i + r_j for all i<j
    # 3. Perimeter <= 4 (nonlinear, handled via penalty or as constraint)
    
    # For SLSQP, we'll use inequality constraints in the form c(x) >= 0
    # But SLSQP handles c(x) >= 0 as constraints.
    
    # Actually, let's use a penalty method with L-BFGS-B for speed, then polish with SLSQP.
    
    # First, use L-BFGS-B with penalty
    best_circles = circles0.copy()
    best_obj = _penalty_objective(best_circles, 100.0)
    
    for pw in [100, 1000, 10000]:
        res = minimize(
            lambda x: _penalty_objective(x, pw),
            best_circles,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_circles = res.x.copy()
    
    # Scale to perimeter 4
    best_circles = _scale_to_perimeter(best_circles)
    
    # Now polish with SLSQP using explicit constraints
    # Variables: flattened array of length 3N
    
    def obj_func(x):
        c = x.reshape(N, 3)
        return -np.sum(c[:, 2])
    
    constraints = []
    
    # Non-negativity of radii
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i + 2]})
    
    # Distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
            })
    
    # Perimeter constraint: 2(w+h) <= 4 => w+h <= 2
    # This is non-smooth, so we'll use a penalty for it in SLSQP or handle separately.
    # Let's add it as a penalty in the objective for SLSQP.
    
    def obj_func_pen(x):
        c = x.reshape(N, 3)
        x_min = np.min(c[:, 0] - c[:, 2])
        x_max = np.max(c[:, 0] + c[:, 2])
        y_min = np.min(c[:, 1] - c[:, 2])
        y_max = np.max(c[:, 1] + c[:, 2])
        w = x_max - x_min
        h = y_max - y_min
        perim = 2 * (w + h)
        penalty = 0.0
        if perim > 4:
            penalty = 100 * (perim - 4)**2
        return -np.sum(c[:, 2]) + penalty
    
    res = minimize(
        obj_func_pen,
        best_circles.flatten(),
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    
    if res.fun < best_obj:
        best_circles = res.x.reshape(N, 3)
        best_circles = _scale_to_perimeter(best_circles)
    
    return best_circles


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_circles = None
    best_sum = 0.0
    
    # Try multiple initializations
    inits = [_init_grid(), _init_random()]
    
    # Add some perturbed grid inits
    for _ in range(5):
        c = _init_grid().copy()
        c[:, 0] += np.random.uniform(-0.01, 0.01, N)
        c[:, 1] += np.random.uniform(-0.01, 0.01, N)
        inits.append(c)
    
    for init in inits:
        if _remaining_time() < 20:
            break
        
        # Scale initial config to perimeter 4
        init_scaled = _scale_to_perimeter(init)
        
        # Solve
        result = _solve_constrained(init_scaled)
        
        # Validate and compute score
        if _validate(result):
            s = np.sum(result[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = result.copy()
    
    # Final polish with tighter tolerance
    if best_circles is not None and _remaining_time() > 5:
        # One more SLSQP pass with tighter ftol
        def obj_func_pen(x):
            c = x.reshape(N, 3)
            x_min = np.min(c[:, 0] - c[:, 2])
            x_max = np.max(c[:, 0] + c[:, 2])
            y_min = np.min(c[:, 1] - c[:, 2])
            y_max = np.max(c[:, 1] + c[:, 2])
            w = x_max - x_min
            h = y_max - y_min
            perim = 2 * (w + h)
            penalty = 0.0
            if perim > 4:
                penalty = 1000 * (perim - 4)**2
            return -np.sum(c[:, 2]) + penalty
        
        constraints = []
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i + 2]})
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
                })
        
        res = minimize(
            obj_func_pen,
            best_circles.flatten(),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        
        if res.success:
            polished = res.x.reshape(N, 3)
            polished = _scale_to_perimeter(polished)
            if _validate(polished):
                s = np.sum(polished[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = polished
    
    if best_circles is None:
        # Fallback
        best_circles = _init_grid()
        best_circles = _scale_to_perimeter(best_circles)
    
    return best_circles
