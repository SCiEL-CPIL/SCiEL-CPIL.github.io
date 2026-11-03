import numpy as np
import time
from scipy.optimize import minimize

N = 26
START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return max(0.0, TIME_LIMIT - _elapsed())

def _softplus(x, beta=1.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 30 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _sigmoid(x, beta=1.0):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1 / (1 + np.exp(-beta * x)), np.exp(beta * x) / (1 + np.exp(beta * x)))

def _validate(sol):
    """Check if solution is valid. Returns True if valid."""
    if sol is None:
        return False
    x = sol[:, 0]
    y = sol[:, 1]
    r = sol[:, 2]
    tol = 1e-9
    for i in range(len(x)):
        if r[i] < -tol:
            return False
        if r[i] > x[i] + tol or r[i] > 1 - x[i] + tol or r[i] > y[i] + tol or r[i] > 1 - y[i] + tol:
            return False
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if r[i] + r[j] > dist + tol:
                return False
    return True

def _enforce_validity(sol, shrink=0.9999):
    """Ensure solution is strictly valid by shrinking radii if needed."""
    if sol is None:
        return None
    x = sol[:, 0].copy()
    y = sol[:, 1].copy()
    r = sol[:, 2].copy()
    
    # Fix boundary violations
    for i in range(len(x)):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i]) * shrink
        r[i] = min(r[i], max_r)
    
    # Fix overlap violations
    for _ in range(20):
        changed = False
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < 1e-12:
                    continue
                if r[i] + r[j] > dist * shrink:
                    excess = (r[i] + r[j]) / dist
                    scale = shrink / excess
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        if not changed:
            break
    
    # Ensure non-negative
    r = np.maximum(r, 0)
    return np.column_stack([x, y, r])

def _score(sol):
    """Return sum of radii if valid, else 0."""
    if sol is None:
        return 0.0
    if not _validate(sol):
        return 0.0
    return float(np.sum(sol[:, 2]))

def _objective_grad(z, beta=100.0):
    """
    Objective: maximize sum(r) subject to constraints.
    Use soft constraints: minimize -sum(r) + penalty * softplus(violations)
    Returns (f, g).
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    
    # Boundary violations: r - x, r - (1-x), r - y, r - (1-y)
    v1 = r - x
    v2 = r - (1 - x)
    v3 = r - y
    v4 = r - (1 - y)
    
    # Overlap violations: (r_i + r_j) - dist_ij
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    overlap = sum_r - dist
    np.fill_diagonal(overlap, 0)
    # Only positive overlaps matter
    overlap_pos = np.maximum(0, overlap)
    
    # Soft penalty
    pen_bound = _softplus(v1, beta) + _softplus(v2, beta) + _softplus(v3, beta) + _softplus(v4, beta)
    pen_overlap = _softplus(overlap_pos, beta)
    
    f = -np.sum(r) + np.sum(pen_bound) + np.sum(pen_overlap)
    
    # Gradient
    g = np.zeros(3 * N)
    g[2*N:3*N] -= 1.0  # d(-sum(r))/dr
    
    # Boundary gradients
    s1 = _sigmoid(v1, beta)
    s2 = _sigmoid(v2, beta)
    s3 = _sigmoid(v3, beta)
    s4 = _sigmoid(v4, beta)
    
    g[0:N] += -s1 + s2  # d(v1)/dx = -1, d(v2)/dx = +1
    g[N:2*N] += -s3 + s4
    g[2*N:3*N] += s1 + s2 + s3 + s4
    
    # Overlap gradients
    safe_dist = np.where(dist > 1e-12, dist, 1e-12)
    # d(overlap)/dx[i] = (x[i]-x[j])/dist for j!=i, only where overlap > 0
    mask = overlap > 0
    s_overlap = _sigmoid(overlap, beta) * mask  # sigmoid of raw overlap, masked
    
    # For x: d/dx[i] sum_j softplus(overlap_ij) = sum_j sigmoid(overlap_ij) * (x[i]-x[j])/dist_ij
    grad_x = np.sum(s_overlap * (dx / safe_dist), axis=1)
    grad_y = np.sum(s_overlap * (dy / safe_dist), axis=1)
    grad_r = np.sum(s_overlap, axis=1)
    
    g[0:N] += grad_x
    g[N:2*N] += grad_y
    g[2*N:3*N] += grad_r
    
    return f, g

def _physics_relax(x, y, r, n_steps=500, dt=0.001, beta=50.0):
    """Physics-based relaxation: push overlapping circles apart."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for step in range(n_steps):
        if _remaining() < 2:
            break
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dx, dy = 0.01, 0.01
                    dist = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    force = beta * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += beta * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= beta * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += beta * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= beta * (y[i] - (1 - r[i]))
        
        x += dt * fx
        y += dt * fy
        
        # Clamp to boundaries
        for i in range(N):
            x[i] = np.clip(x[i], r[i] * 0.99, 1 - r[i] * 0.99)
            y[i] = np.clip(y[i], r[i] * 0.99, 1 - r[i] * 0.99)
    
    return x, y, r

def _polish_analytic(x, y, r, n_iters=100):
    """Iteratively adjust radii to maximize sum while maintaining constraints."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(n_iters):
        if _remaining() < 2:
            break
        changed = False
        for i in range(N):
            # Max possible radius for circle i given positions
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            for j in range(N):
                if i == j:
                    continue
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                max_r = min(max_r, (dist - r[j]))
            if max_r > r[i] * (1 + 1e-6):
                r[i] = max_r * 0.9999
                changed = True
        if not changed:
            break
    
    return x, y, r

def _initial_config_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    while len(pts) < N:
        pts.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    pts = np.array(pts[:N])
    # Compute initial radii
    r = np.zeros(N)
    for i in range(N):
        min_d = min(pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
        for j in range(N):
            if i != j:
                min_d = min(min_d, np.linalg.norm(pts[i] - pts[j]))
        r[i] = min_d * 0.45
    return pts[:, 0], pts[:, 1], r

def _initial_config_hex():
    """Hexagonal lattice initialization."""
    pts = []
    n_per_row = 5
    for i in range(n_per_row):
        for j in range(6):
            x = (j + 0.5) / 6.0 + (0.5 if i % 2 == 1 else 0) / 6.0
            y = (i + 0.5) / 5.0
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x + np.random.uniform(-0.02, 0.02), y + np.random.uniform(-0.02, 0.02)])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    while len(pts) < N:
        pts.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    pts = np.array(pts[:N])
    r = np.zeros(N)
    for i in range(N):
        min_d = min(pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
        for j in range(N):
            if i != j:
                min_d = min(min_d, np.linalg.norm(pts[i] - pts[j]))
        r[i] = min_d * 0.45
    return pts[:, 0], pts[:, 1], r

def _initial_config_voronoi():
    """Voronoi-like: grid then relax positions."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    while len(pts) < N:
        pts.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    pts = np.array(pts[:N])
    
    # Relax: move each point away from nearest neighbor
    for _ in range(100):
        for i in range(N):
            min_dist = np.inf
            nearest = -1
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
                    nearest = j
            if nearest >= 0 and min_dist < 0.25:
                direction = pts[i] - pts[nearest]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    pts[i] += 0.01 * direction
                    pts[i] = np.clip(pts[i], 0.05, 0.95)
    
    r = np.zeros(N)
    for i in range(N):
        min_d = min(pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
        for j in range(N):
            if i != j:
                min_d = min(min_d, np.linalg.norm(pts[i] - pts[j]))
        r[i] = min_d * 0.45
    return pts[:, 0], pts[:, 1], r

def _optimize_interior_point(x0, y0, r0, max_time=15.0):
    """Interior-point style optimization with increasing beta."""
    start = time.time()
    x, y, r = x0.copy(), y0.copy(), r0.copy()
    
    betas = [10, 20, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() - start > max_time or _remaining() < 5:
            break
        
        z0 = np.concatenate([x, y, r])
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        try:
            res = minimize(
                lambda z: _objective_grad(z, beta)[0],
                z0,
                jac=lambda z: _objective_grad(z, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < _objective_grad(z0, beta)[0]:
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
        except Exception:
            pass
        
        # Physics relaxation step
        if _remaining() < 5:
            break
        x, y, r = _physics_relax(x, y, r, n_steps=200, dt=0.002, beta=beta)
        # Polish
        x, y, r = _polish_analytic(x, y, r, n_iters=50)
    
    return x, y, r

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_sol = None
    best_score = 0.0
    
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    # Multiple initializations
    inits = []
    inits.append(_initial_config_grid())
    inits.append(_initial_config_hex())
    inits.append(_initial_config_voronoi())
    
    # Random inits
    for _ in range(3):
        pts = np.random.uniform(0.1, 0.9, size=(N, 2))
        r = np.zeros(N)
        for i in range(N):
            min_d = min(pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
            for j in range(N):
                if i != j:
                    min_d = min(min_d, np.linalg.norm(pts[i] - pts[j]))
            r[i] = min_d * 0.4
        inits.append((pts[:, 0], pts[:, 1], r))
    
    # Optimize from each init
    for x0, y0, r0 in inits:
        if _remaining() < 8:
            break
        
        x, y, r = _optimize_interior_point(x0, y0, r0, max_time=min(12.0, _remaining() - 3))
        sol = np.column_stack([x, y, r])
        sol = _enforce_validity(sol, shrink=0.9999)
        score = _score(sol)
        if score > best_score:
            best_score = score
            best_sol = sol
    
    # Local refinement from best
    if best_sol is not None and _remaining() > 10:
        for restart in range(4):
            if _remaining() < 8:
                break
            x = best_sol[:, 0] + np.random.normal(0, 0.015, size=N)
            y = best_sol[:, 1] + np.random.normal(0, 0.015, size=N)
            r = best_sol[:, 2] * 0.92
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            
            x, y, r = _optimize_interior_point(x, y, r, max_time=min(10.0, _remaining() - 3))
            sol = np.column_stack([x, y, r])
            sol = _enforce_validity(sol, shrink=0.9999)
            score = _score(sol)
            if score > best_score:
                best_score = score
                best_sol = sol
    
    # Fallback
    if best_sol is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        best_sol = np.array(cells[:N])
    
    return best_sol
