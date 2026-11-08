"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Precompute coefficient matrices for cross product:
# cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
# = -xi*yk + xi*yi + xj*yi - xj*yk + yi*xk - yi*xi + yj*xi - yj*xk + yk*xj - yk*xi + yk*xi - yi*xk
# Let's just compute gradient numerically or use the formula directly.
# For point p_a, the cross product of triangles involving p_a:
# We'll compute gradients numerically for simplicity and robustness.


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas. pts has shape (N, 2). Returns array of shape (165,)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _soft_min_area(pts, temperature=0.001):
    """Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and helps with gradient-based optimization."""
    areas = _all_triangle_areas(pts)
    # Soft minimum: -log-sum-exp(-areas/T) * T
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -lse * temperature


def _soft_min_area_grad(pts, temperature=0.001):
    """Compute soft-min area and its gradient w.r.t. points.
    Returns (soft_min_value, grad) where grad has shape (N, 2)."""
    areas = _all_triangle_areas(pts)
    # Soft minimum: f = -T * log(sum(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    f = -(max_scaled + np.log(sum_exp)) * temperature
    
    # Gradient of f w.r.t. areas: df/darea_i = (exp(-area_i/T) / sum(exp(-area_j/T)))
    # = exp_scaled[i] / sum_exp
    weights = exp_scaled / sum_exp  # shape (165,)
    
    # Now need d(area_i)/d(pts)
    # area_i = 0.5 * |cross_i| where cross_i = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # d(area_i)/d(pts_a) for point a in {i,j,k}
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # dcross/dxi = -(yk-yi) + (yj-yi)  = yj - yk  ... wait let me redo
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # dcross/dxi = -(yk-yi) + (yj-yi) = yj - yk
    # dcross/dxj = (yk-yi)
    # dcross/dyi = -(xj-xi) - (xk-xi) = -xj + xi - xk + xi = 2*xi - xj - xk
    # dcross/dyj = -(xk-xi)
    # dcross/dxk = -(yj-yi)
    # dcross/dyk = (xj-xi)
    
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    
    xi, yi = pi[:, 0], pi[:, 1]
    xj, yj = pj[:, 0], pj[:, 1]
    xk, yk = pk[:, 0], pk[:, 1]
    
    # d(cross)/d(each point coordinate)
    dxi = yj - yk
    dxi_x = dxi  # dcross/dxi
    dxi_y = 2*xi - xj - xk  # dcross/dyi
    dxj_x = yk - yi  # dcross/dxj
    dxj_y = -(xk - xi)  # dcross/dyj
    dxk_x = -(yj - yi)  # dcross/dxk
    dxk_y = xj - xi  # dcross/dyk
    
    # area = 0.5 * |cross|, so darea/dvar = 0.5 * sign(cross) * dcross/dvar
    cross_vals = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    sign_cross = np.sign(cross_vals)
    
    # Gradient contribution per triangle: 0.5 * sign * dcross/dvar * weight
    w = 0.5 * weights * sign_cross  # shape (165,)
    
    grad = np.zeros((N, 2))
    # Point i contributions
    np.add.at(grad[:, 0], _idx_i, w * dxi_x)
    np.add.at(grad[:, 1], _idx_i, w * dxi_y)
    # Point j contributions
    np.add.at(grad[:, 0], _idx_j, w * dxj_x)
    np.add.at(grad[:, 1], _idx_j, w * dxj_y)
    # Point k contributions
    np.add.at(grad[:, 0], _idx_k, w * dxk_x)
    np.add.at(grad[:, 1], _idx_k, w * dxk_y)
    
    return f, grad


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Constraints:
      y >= 0
      y <= sqrt(3) * x        (left edge from origin to top)
      y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)  (right edge)
    """
    pts = pts.copy()
    # Clip y >= 0
    pts[pts[:, 1] < 0, 1] = 0.0
    # Clip to left edge: y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    min_x_left = pts[:, 1] / np.sqrt(3)
    mask_left = pts[:, 0] < min_x_left
    pts[mask_left, 0] = min_x_left[mask_left]
    # Clip to right edge: y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    max_x_right = 1.0 - pts[:, 1] / np.sqrt(3)
    mask_right = pts[:, 0] > max_x_right
    pts[mask_right, 0] = max_x_right[mask_right]
    # Final clip for numerical safety
    pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1)
    return pts


def _project_grad(pts, grad):
    """Project points and adjust gradient to account for projection (zero out components
    that would move points outside the triangle)."""
    pts_new = _project_to_triangle(pts)
    # For simplicity, just zero out gradient components where projection occurred
    # This is a first-order approximation
    grad_new = grad.copy()
    # Where y was clipped to 0, zero out y-gradient
    grad_new[pts[:, 1] < 0, 1] = 0.0
    # Where x was clipped on left edge, zero out x-gradient
    min_x_left = pts[:, 1] / np.sqrt(3)
    mask_left = pts[:, 0] < min_x_left
    grad_new[mask_left, 0] = 0.0
    # Where x was clipped on right edge, zero out x-gradient
    max_x_right = 1.0 - pts[:, 1] / np.sqrt(3)
    mask_right = pts[:, 0] > max_x_right
    grad_new[mask_right, 0] = 0.0
    return pts_new, grad_new


def _objective_smooth(x, temperature=0.001):
    """Negative soft-min triangle area (smooth, to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_soft_min_area(pts, temperature)


def _objective_exact(x):
    """Negative exact min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Create a symmetric initial configuration for 11 points."""
    cx, cy = 0.5, _H / 3
    r = 0.25
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [[cx, cy]]
    for a in angles:
        px = cx + r * np.cos(a)
        py = cy + r * np.sin(a)
        pts.append([px, py])
    pts.append([0.15, 0.1])
    pts.append([0.85, 0.1])
    pts.append([0.5, 0.7])
    pts.append([0.35, 0.35])
    arr = np.array(pts[:N], dtype=float)
    return _project_to_triangle(arr).flatten()


def _random_initial_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        while u + v > 1.0:
            u, v = rng.random(), rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    arr = np.array(pts, dtype=float)
    return _project_to_triangle(arr).flatten()


def _nelder_mead_optimize(x0, max_time=15.0, temperature=0.001):
    """Run Nelder-Mead optimization with time-based stopping.
    Returns (best_x, best_val_exact)."""
    start = time.time()
    
    # Use smooth objective for Nelder-Mead
    def obj(x):
        return _objective_smooth(x, temperature)
    
    best_x = x0.copy()
    best_val_smooth = obj(best_x)
    best_val_exact = _objective_exact(best_x)
    
    def callback(xk):
        nonlocal best_x, best_val_smooth, best_val_exact
        val_s = obj(xk)
        if val_s < best_val_smooth:
            best_x = xk.copy()
            best_val_smooth = val_s
            best_val_exact = _objective_exact(xk)
        return time.time() - start > max_time
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'callback': callback}
        )
        if res.fun < best_val_smooth:
            best_x = res.x
            best_val_smooth = res.fun
            best_val_exact = _objective_exact(best_x)
    except Exception:
        pass
    
    return best_x, best_val_exact


def _gradient_ascent_optimize(x0, max_time=30.0):
    """Projected gradient ascent on soft-min area with adaptive step size and
    temperature annealing. Returns (best_x, best_val_exact)."""
    start = time.time()
    pts = _project_to_triangle(x0.reshape(N, 2)).copy()
    
    best_val = _min_triangle_area(pts)
    best_pts = pts.copy()
    
    # Adaptive parameters
    lr = 0.005  # initial learning rate
    temperature = 0.005  # initial soft-min temperature
    min_temp = 0.0001
    temp_decay = 0.995
    lr_decay = 0.999
    lr_min = 0.0001
    
    # Backtracking line search parameters
    c1 = 1e-4  # Armijo condition
    
    while time.time() - start < max_time:
        # Compute soft-min value and gradient
        val, grad = _soft_min_area_grad(pts, temperature)
        
        # Project gradient
        pts_proj, grad_proj = _project_grad(pts, grad)
        
        # Gradient ascent step
        new_pts = pts + lr * grad_proj
        new_pts = _project_to_triangle(new_pts)
        
        # Backtracking line search
        new_val = _min_triangle_area(new_pts)
        if new_val <= val:
            # Try a few backtracking steps
            for backtrack in range(5):
                trial_pts = pts + lr * (0.5 ** (backtrack + 1)) * grad_proj
                trial_pts = _project_to_triangle(trial_pts)
                trial_val = _min_triangle_area(trial_pts)
                if trial_val > val:
                    new_pts = trial_pts
                    new_val = trial_val
                    break
                else:
                    new_pts = trial_pts
                    new_val = trial_val
            else:
                # If no improvement, reduce learning rate
                lr *= 0.5
        
        # Accept if improved
        if new_val > best_val:
            best_val = new_val
            best_pts = new_pts.copy()
        
        pts = new_pts
        
        # Anneal temperature and learning rate
        temperature = max(min_temp, temperature * temp_decay)
        lr = max(lr_min, lr * lr_decay)
        
        # Check if we're stuck
        if abs(new_val - val) < 1e-10:
            # Add small perturbation
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            pts = pts + rng.randn(N, 2) * 0.001
            pts = _project_to_triangle(pts)
    
    return best_pts.flatten(), best_val


def _slsqp_polish(x0, max_iter=300):
    """Final SLSQP polish with exact objective."""
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _objective_exact(best_x)
    
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
        )
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    return best_x, best_val


def _coordinate_descent_polish(x0, max_time=10.0):
    """Coordinate descent polish: cycle through each point and optimize it
    while keeping others fixed. Returns (best_x, best_val_exact)."""
    start = time.time()
    pts = _project_to_triangle(x0.reshape(N, 2)).copy()
    best_val = _min_triangle_area(pts)
    best_pts = pts.copy()
    
    # For each point, try small perturbations in x and y directions
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    while time.time() - start < max_time:
        improved = False
        for i in range(N):
            if time.time() - start > max_time:
                break
            for step in step_sizes:
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (-step, -step), (step, -step), (-step, step)]:
                    trial_pts = pts.copy()
                    trial_pts[i, 0] += dx
                    trial_pts[i, 1] += dy
                    trial_pts = _project_to_triangle(trial_pts)
                    trial_val = _min_triangle_area(trial_pts)
                    if trial_val > best_val:
                        best_val = trial_val
                        best_pts = trial_pts.copy()
                        pts = trial_pts.copy()
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_pts.flatten(), best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_x = None
    best_val = np.inf
    
    # Start 1: symmetric configuration - Nelder-Mead
    x0 = _symmetric_initial_config()
    x, val = _nelder_mead_optimize(x0, max_time=12.0, temperature=0.001)
    best_x, best_val = x, val
    
    # Gradient ascent from same start
    if time.time() < deadline - 60:
        x, val = _gradient_ascent_optimize(x0, max_time=20.0)
        if val > best_val:
            best_x, best_val = x, val
    
    # Additional random starts with Nelder-Mead on smooth objective
    n_starts = 20
    for seed in range(n_starts):
        if time.time() > deadline - 35:
            break
        x0 = _random_initial_config(seed + 100)
        remaining = deadline - time.time() - 35
        if remaining < 3:
            break
        tm = min(8.0, remaining / 2)
        x, val = _nelder_mead_optimize(x0, max_time=tm, temperature=0.001)
        if val > best_val:
            best_x, best_val = x, val
    
    # Gradient ascent from best solution
    if time.time() < deadline - 15:
        x, val = _gradient_ascent_optimize(best_x, max_time=min(10.0, deadline - time.time() - 15))
        if val > best_val:
            best_x, best_val = x, val
    
    # Coordinate descent polish
    if time.time() < deadline - 8:
        x, val = _coordinate_descent_polish(best_x, max_time=min(8.0, deadline - time.time() - 8))
        if val > best_val:
            best_x, best_val = x, val
    
    # Final refinement: polish with SLSQP on exact objective
    if time.time() < deadline - 5:
        best_x, best_val = _slsqp_polish(best_x, max_iter=300)
    
    # One more polish if time permits
    if time.time() < deadline - 2:
        best_x, best_val = _slsqp_polish(best_x, max_iter=200)
    
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts
