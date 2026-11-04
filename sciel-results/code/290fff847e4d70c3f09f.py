"""Circle packing in perimeter-4 rectangle - force-directed + SLSQP hard constraints."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _check_valid(circles, tol=1e-7):
    """Check if circles are valid (disjoint, positive radii, perimeter <= 4)."""
    if len(circles) != N:
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(rs <= 0):
        return False
    xmin, xmax = np.min(xs - rs), np.max(xs + rs)
    ymin, ymax = np.min(ys - rs), np.max(ys + rs)
    if 2 * ((xmax - xmin) + (ymax - ymin)) > 4.0 + 1e-6:
        return False
    # Check disjoint interiors
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    req = rs[:, None] + rs[None, :]
    mask = ~np.eye(N, dtype=bool)
    if np.any(dist[mask] < req[mask] - tol):
        return False
    return True

def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))

def _force_directed_init(n, w, h, seed=42, iters=300):
    """Force-directed layout to get a good initial packing."""
    rng = np.random.default_rng(seed)
    # Start with small circles in a rough grid
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    cw, ch = w / cols, h / rows
    xs = np.zeros(n)
    ys = np.zeros(n)
    rs = np.zeros(n)
    for i in range(n):
        r_i = i // cols
        c_i = i % cols
        xs[i] = (c_i + 0.5) * cw
        ys[i] = (r_i + 0.5) * ch
        rs[i] = 0.5 * min(cw, ch) * 0.8
    
    # Force-directed iterations
    for it in range(iters):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Repulsive forces between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-10:
                    dx, dy, d = 0.01, 0.01, 0.01414
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    force = 10.0 * overlap
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d
        # Attractive force to keep circles within bounds
        for i in range(n):
            if xs[i] - rs[i] < 0:
                fx[i] += 5.0 * (0 - (xs[i] - rs[i]))
            if xs[i] + rs[i] > w:
                fx[i] -= 5.0 * ((xs[i] + rs[i]) - w)
            if ys[i] - rs[i] < 0:
                fy[i] += 5.0 * (0 - (ys[i] - rs[i]))
            if ys[i] + rs[i] > h:
                fy[i] -= 5.0 * ((ys[i] + rs[i]) - h)
        # Update positions with damping
        lr = 0.3 * (1.0 - it / iters) + 0.05
        xs += lr * fx
        ys += lr * fy
        # Clamp to bounds
        xs = np.clip(xs, rs, w - rs)
        ys = np.clip(ys, rs, h - rs)
    
    return np.column_stack([xs, ys, rs])

def _slsqp_optimize(circles, w, h, max_iter=200):
    """SLSQP optimization with hard constraints."""
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(x):
        xs, ys, rs = x[:n], x[n:2*n], x[2*n:]
        return -np.sum(rs)
    
    # Constraints: non-overlap
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(i, j):
                def constraint(x):
                    xs, ys, rs = x[:n], x[n:2*n], x[2*n:]
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    return np.sqrt(dx**2 + dy**2) - rs[i] - rs[j]
                constraint.name = f'nonoverlap_{i}_{j}'
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Perimeter constraint
    def perimeter_constraint(x):
        xs, ys, rs = x[:n], x[n:2*n], x[2*n:]
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        return 4.0 - 2 * ((xmax - xmin) + (ymax - ymin))
    constraints.append({'type': 'ineq', 'fun': perimeter_constraint})
    
    # Bounds: radii > 0, positions reasonable
    bounds = []
    for i in range(n):
        bounds.append((0, w))  # x
    for i in range(n):
        bounds.append((0, h))  # y
    for i in range(n):
        bounds.append((1e-6, max(w, h)))  # r
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        return result.x.reshape(3, n).T
    except Exception:
        return circles

def _polish(circles, w, h, max_iter=100):
    """High-precision polishing with tight tolerances."""
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(x):
        xs, ys, rs = x[:n], x[n:2*n], x[2*n:]
        return -np.sum(rs)
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(i, j):
                def constraint(x):
                    xs, ys, rs = x[:n], x[n:2*n], x[2*n:]
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    return np.sqrt(dx**2 + dy**2) - rs[i] - rs[j]
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    def perimeter_constraint(x):
        xs, ys, rs = x[:n], x[n:2*n], x[2*n:]
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        return 4.0 - 2 * ((xmax - xmin) + (ymax - ymin))
    constraints.append({'type': 'ineq', 'fun': perimeter_constraint})
    
    bounds = []
    for i in range(n):
        bounds.append((0, w))
    for i in range(n):
        bounds.append((0, h))
    for i in range(n):
        bounds.append((1e-6, max(w, h)))
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        return result.x.reshape(3, n).T
    except Exception:
        return circles

def _try_aspect(w, h, seed=0, deadline=None):
    """Try to pack circles in a rectangle of given dimensions."""
    if deadline and time.time() > deadline:
        return None
    
    # Force-directed initialization
    circles = _force_directed_init(N, w, h, seed=seed, iters=200)
    
    # SLSQP optimization
    circles = _slsqp_optimize(circles, w, h, max_iter=150)
    
    # Polish
    circles = _polish(circles, w, h, max_iter=80)
    
    if _check_valid(circles):
        return circles
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_circles = None
    best_value = 0.0
    
    # Try different aspect ratios: w + h = 2, so h = 2 - w
    # Perimeter = 2(w+h) = 4
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4286, 0.75, 1.3333, 0.6, 1.6667, 0.9, 1.1111]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 / (1.0 + aspect)
        h = aspect * w
        
        for seed in range(3):
            if time.time() > deadline:
                break
            
            circles = _try_aspect(w, h, seed=seed, deadline=deadline)
            if circles is not None:
                val = _sum_radii(circles)
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
    
    # Fallback: simple grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([0.5 + j * 1.0/side, 0.5 + i * 1.0/side, 0.5/side - 1e-4])
        best_circles = np.array(cells[:N])
    
    return best_circles
