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
M = len(_combos)  # 165


def _all_triangle_areas(pts):
    """Compute all 165 triangle areas. pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    return np.min(_all_triangle_areas(pts))


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Constraints:
      y >= 0
      y <= sqrt(3) * x        (left edge from origin to top)
      y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)  (right edge)
    """
    pts = pts.copy()
    pts[pts[:, 1] < 0, 1] = 0.0
    min_x_left = pts[:, 1] / np.sqrt(3)
    mask_left = pts[:, 0] < min_x_left
    pts[mask_left, 0] = min_x_left[mask_left]
    max_x_right = 1.0 - pts[:, 1] / np.sqrt(3)
    mask_right = pts[:, 0] > max_x_right
    pts[mask_right, 0] = max_x_right[mask_right]
    pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1)
    return pts


def _cross_and_grad(pts):
    """Compute cross products and their gradients w.r.t. each point.
    Returns (cross_vals, grad_x, grad_y) each of shape (M,) for cross,
    and (N, 2) for the full gradient of the signed area sum.
    
    cross_m = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    d(cross)/dx_i = yj - yk
    d(cross)/dy_i = 2*xi - xj - xk
    d(cross)/dx_j = yk - yi
    d(cross)/dy_j = -(xk - xi)
    d(cross)/dx_k = -(yj - yi)
    d(cross)/dy_k = xj - xi
    """
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    
    xi, yi = pi[:, 0], pi[:, 1]
    xj, yj = pj[:, 0], pj[:, 1]
    xk, yk = pk[:, 0], pk[:, 1]
    
    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    
    # Gradient of cross w.r.t. each point's coordinates
    gxi = yj - yk
    gyi = 2*xi - xj - xk
    gxj = yk - yi
    gyj = -(xk - xi)
    gxk = -(yj - yi)
    gyk = xj - xi
    
    return cross, (gxi, gyi, gxj, gyj, gxk, gyk)


def _qcqp_objective_and_grad(pts, t):
    """Epigraph formulation: maximize t subject to all areas >= t.
    We minimize a penalty: sum of max(0, t - area_m)^2 for all m.
    This is a smooth (C^1) function when areas > t, and has kinks at area_m = t.
    
    Returns (penalty_value, gradient w.r.t. pts).
    """
    areas = _all_triangle_areas(pts)
    cross, grads = _cross_and_grad(pts)
    
    # deficit_m = max(0, t - area_m)
    deficit = np.maximum(0.0, t - areas)
    penalty = np.sum(deficit ** 2)
    
    # Gradient of penalty w.r.t. pts:
    # d(penalty)/d(pts) = sum_m 2*deficit_m * d(t-area_m)/d(pts)
    # = -sum_m 2*deficit_m * d(area_m)/d(pts)
    # area_m = 0.5 * |cross_m|, so d(area_m)/d(pts) = 0.5 * sign(cross_m) * d(cross_m)/d(pts)
    # (when cross_m != 0)
    
    sign_cross = np.sign(cross)
    # For zero cross, use 1 (arbitrary, but measure zero)
    sign_cross[sign_cross == 0] = 1.0
    
    # Only triangles with deficit > 0 contribute
    active = deficit > 0
    if not np.any(active):
        return penalty, np.zeros((N, 2))
    
    w = 2.0 * deficit[active] * 0.5 * sign_cross[active]  # shape (n_active,)
    w = -w  # because d(t-area)/d(pts) = -d(area)/d(pts)
    
    gxi, gyi, gxj, gyj, gxk, gyk = grads
    gxi_a, gyi_a = gxi[active], gyi[active]
    gxj_a, gyj_a = gxj[active], gyj[active]
    gxk_a, gyk_a = gxk[active], gyk[active]
    
    idx_i_a = _idx_i[active]
    idx_j_a = _idx_j[active]
    idx_k_a = _idx_k[active]
    
    grad = np.zeros((N, 2))
    np.add.at(grad[:, 0], idx_i_a, w * gxi_a)
    np.add.at(grad[:, 1], idx_i_a, w * gyi_a)
    np.add.at(grad[:, 0], idx_j_a, w * gxj_a)
    np.add.at(grad[:, 1], idx_j_a, w * gyj_a)
    np.add.at(grad[:, 0], idx_k_a, w * gxk_a)
    np.add.at(grad[:, 1], idx_k_a, w * gyk_a)
    
    return penalty, grad


def _soft_min_area(pts, temperature=0.002):
    """Smooth approximation of min triangle area via log-sum-exp."""
    areas = _all_triangle_areas(pts)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -lse * temperature


def _soft_min_grad(pts, temperature=0.002):
    """Gradient of soft-min area. Returns (value, grad)."""
    areas = _all_triangle_areas(pts)
    cross, grads = _cross_and_grad(pts)
    
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    f = -(max_scaled + np.log(sum_exp)) * temperature
    
    weights = exp_scaled / sum_exp  # shape (M,)
    
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    
    w = 0.5 * weights * sign_cross  # d(area_m)/d(cross_m) = 0.5*sign(cross_m)
    
    gxi, gyi, gxj, gyj, gxk, gyk = grads
    
    grad = np.zeros((N, 2))
    np.add.at(grad[:, 0], _idx_i, w * gxi)
    np.add.at(grad[:, 1], _idx_i, w * gyi)
    np.add.at(grad[:, 0], _idx_j, w * gxj)
    np.add.at(grad[:, 1], _idx_j, w * gyj)
    np.add.at(grad[:, 0], _idx_k, w * gxk)
    np.add.at(grad[:, 1], _idx_k, w * gyk)
    
    return f, grad


def _nelder_mead_optimize(x0, max_time=15.0, temperature=0.002):
    """Nelder-Mead on smooth objective."""
    start = time.time()
    
    def obj(x):
        pts = _project_to_triangle(x.reshape(N, 2))
        return -_soft_min_area(pts, temperature)
    
    best_x = x0.copy()
    best_val_smooth = obj(best_x)
    best_val_exact = _min_triangle_area(_project_to_triangle(best_x.reshape(N, 2)))
    
    def callback(xk):
        nonlocal best_x, best_val_smooth, best_val_exact
        val_s = obj(xk)
        if val_s < best_val_smooth:
            best_x = xk.copy()
            best_val_smooth = val_s
            best_val_exact = _min_triangle_area(_project_to_triangle(best_x.reshape(N, 2)))
        return time.time() - start > max_time
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'callback': callback}
        )
        if res.fun < best_val_smooth:
            best_x = res.x
            best_val_smooth = res.fun
            best_val_exact = _min_triangle_area(_project_to_triangle(best_x.reshape(N, 2)))
    except Exception:
        pass
    
    return best_x, best_val_exact


def _qcqp_ascent(pts, max_time=20.0):
    """Projected gradient ascent using QCQP epigraph formulation.
    We jointly optimize point positions and t (the min area target).
    
    Strategy: 
    - Maintain t as the current min area (or slightly below)
    - At each step: compute penalty gradient w.r.t. pts, take a step
    - Update t = min(current_t + small_increment, new_min_area)
    - Use backtracking line search on the actual min area
    """
    start = time.time()
    pts = _project_to_triangle(pts.copy())
    
    current_t = _min_triangle_area(pts)
    best_val = current_t
    best_pts = pts.copy()
    
    lr = 0.003
    t_increment = 0.0001
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start < max_time:
        # Compute penalty and gradient
        penalty, grad = _qcqp_objective_and_grad(pts, current_t)
        
        # Gradient ascent step (we want to reduce penalty, which means increasing areas)
        # The gradient of penalty w.r.t. pts points in the direction of increasing penalty
        # So we go in the negative gradient direction to reduce penalty
        new_pts = pts - lr * grad
        new_pts = _project_to_triangle(new_pts)
        
        new_area = _min_triangle_area(new_pts)
        
        if new_area > best_val + 1e-12:
            best_val = new_area
            best_pts = new_pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Backtracking
            for bt in range(6):
                trial_lr = lr * (0.5 ** (bt + 1))
                trial_pts = pts - trial_lr * grad
                trial_pts = _project_to_triangle(trial_pts)
                trial_area = _min_triangle_area(trial_pts)
                if trial_area > _min_triangle_area(pts) + 1e-12:
                    new_pts = trial_pts
                    new_area = trial_area
                    if new_area > best_val:
                        best_val = new_area
                        best_pts = new_pts.copy()
                        no_improve_count = 0
                    break
                elif trial_area > _min_triangle_area(pts):
                    new_pts = trial_pts
                    new_area = trial_area
                    break
        
        pts = new_pts
        
        # Update t: increase it slightly toward the current min area
        current_t = min(current_t + t_increment, _min_triangle_area(pts))
        
        # Decay learning rate slowly
        lr = max(0.0001, lr * 0.999)
        
        # If stuck, perturb
        if no_improve_count > max_no_improve:
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            pts = pts + rng.randn(N, 2) * 0.002
            pts = _project_to_triangle(pts)
            current_t = _min_triangle_area(pts)
            no_improve_count = 0
            lr = 0.003
    
    return best_pts.flatten(), best_val


def _soft_min_ascent(pts, max_time=20.0):
    """Projected gradient ascent on soft-min area."""
    start = time.time()
    pts = _project_to_triangle(pts.copy())
    
    best_val = _min_triangle_area(pts)
    best_pts = pts.copy()
    
    lr = 0.004
    temperature = 0.005
    min_temp = 0.0002
    temp_decay = 0.997
    lr_decay = 0.9995
    lr_min = 0.0001
    no_improve = 0
    max_no_improve = 80
    
    while time.time() - start < max_time:
        val, grad = _soft_min_grad(pts, temperature)
        
        new_pts = pts + lr * grad
        new_pts = _project_to_triangle(new_pts)
        
        new_val = _min_triangle_area(new_pts)
        
        if new_val > best_val + 1e-12:
            best_val = new_val
            best_pts = new_pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            for bt in range(6):
                trial_lr = lr * (0.5 ** (bt + 1))
                trial_pts = pts + trial_lr * grad
                trial_pts = _project_to_triangle(trial_pts)
                trial_val = _min_triangle_area(trial_pts)
                if trial_val > _min_triangle_area(pts) + 1e-12:
                    new_pts = trial_pts
                    new_val = trial_val
                    if new_val > best_val:
                        best_val = new_val
                        best_pts = new_pts.copy()
                        no_improve = 0
                    break
                elif trial_val > _min_triangle_area(pts):
                    new_pts = trial_pts
                    new_val = trial_val
                    break
        
        pts = new_pts
        temperature = max(min_temp, temperature * temp_decay)
        lr = max(lr_min, lr * lr_decay)
        
        if no_improve > max_no_improve:
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            pts = pts + rng.randn(N, 2) * 0.002
            pts = _project_to_triangle(pts)
            no_improve = 0
            lr = 0.004
    
    return best_pts.flatten(), best_val


def _coordinate_descent_polish(x0, max_time=10.0):
    """Coordinate descent: optimize one point at a time."""
    start = time.time()
    pts = _project_to_triangle(x0.reshape(N, 2)).copy()
    best_val = _min_triangle_area(pts)
    best_pts = pts.copy()
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
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
                    if trial_val > best_val + 1e-12:
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


def _slsqp_polish(x0, max_iter=300):
    """SLSQP polish."""
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    def obj(x):
        pts = _project_to_triangle(x.reshape(N, 2))
        return -_min_triangle_area(pts)
    
    best_x = x0.copy()
    best_val = obj(best_x)
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    return best_x, -best_val


def _symmetric_initial_config():
    """Create a symmetric initial configuration."""
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


def _structured_initial_config():
    """A more structured initial config based on known good patterns."""
    # Place points in a grid-like pattern within the triangle
    pts = [
        [0.5, 0.02],       # bottom center
        [0.2, 0.02],       # bottom left
        [0.8, 0.02],       # bottom right
        [0.35, 0.2],       # mid-left
        [0.65, 0.2],       # mid-right
        [0.5, 0.35],       # center
        [0.25, 0.4],       # upper-left
        [0.75, 0.4],       # upper-right
        [0.4, 0.55],       # top-left
        [0.6, 0.55],       # top-right
        [0.5, 0.7],        # top
    ]
    arr = np.array(pts, dtype=float)
    return _project_to_triangle(arr).flatten()


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_x = None
    best_val = -np.inf
    
    # Phase 1: Multi-start Nelder-Mead on smooth objective
    starts = [_symmetric_initial_config(), _structured_initial_config()]
    for seed in range(1, 8):
        starts.append(_random_initial_config(seed + 100))
    
    for x0 in starts:
        if time.time() > deadline - 50:
            break
        x, val = _nelder_mead_optimize(x0, max_time=6.0, temperature=0.002)
        if val > best_val:
            best_val = val
            best_x = x
    
    # Phase 2: QCQP epigraph ascent from best solution
    if time.time() < deadline - 35:
        x, val = _qcqp_ascent(best_x, max_time=min(15.0, deadline - time.time() - 35))
        if val > best_val:
            best_val = val
            best_x = x
    
    # Phase 3: Soft-min gradient ascent from best solution
    if time.time() < deadline - 25:
        x, val = _soft_min_ascent(best_x, max_time=min(12.0, deadline - time.time() - 25))
        if val > best_val:
            best_val = val
            best_x = x
    
    # Phase 4: More Nelder-Mead from best with lower temperature
    if time.time() < deadline - 20:
        x, val = _nelder_mead_optimize(best_x, max_time=min(8.0, deadline - time.time() - 20), temperature=0.001)
        if val > best_val:
            best_val = val
            best_x = x
    
    # Phase 5: QCQP ascent again
    if time.time() < deadline - 15:
        x, val = _qcqp_ascent(best_x, max_time=min(10.0, deadline - time.time() - 15))
        if val > best_val:
            best_val = val
            best_x = x
    
    # Phase 6: Coordinate descent polish
    if time.time() < deadline - 8:
        x, val = _coordinate_descent_polish(best_x, max_time=min(7.0, deadline - time.time() - 8))
        if val > best_val:
            best_val = val
            best_x = x
    
    # Phase 7: SLSQP polish
    if time.time() < deadline - 5:
        best_x, best_val = _slsqp_polish(best_x, max_iter=300)
    
    # Phase 8: Final coordinate descent
    if time.time() < deadline - 2:
        x, val = _coordinate_descent_polish(best_x, max_time=min(2.0, deadline - time.time() - 2))
        if val > best_val:
            best_val = val
            best_x = x
    
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts
