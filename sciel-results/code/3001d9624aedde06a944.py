"""Circle packing in perimeter-4 rectangle: Multi-start SLSQP + iterative radius polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _make_constraints(w, h):
    """Create constraint functions for SLSQP."""
    constraints = []
    
    # Boundary constraints: each circle must be inside the rectangle
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*N+i]})  # x_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: (w - z[i]) - z[2*N+i]})  # w - x_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i] - z[2*N+i]})  # y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: (h - z[N+i]) - z[2*N+i]})  # h - y_i >= r_i
    
    # Non-overlap constraints: for each pair (i, j), i < j
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: (z[i] - z[j])**2 + (z[N+i] - z[N+j])**2 - (z[2*N+i] + z[2*N+j])**2
            })
    
    return constraints

def _slsqp_optimize(x0, y0, r0, w, h, time_budget, maxiter=300):
    """Optimize using SLSQP with constraints."""
    start = time.time()
    
    z0 = np.concatenate([x0, y0, r0])
    bounds = [(0, w)] * N + [(0, h)] * N + [(0, min(w, h) / 2)] * N
    constraints = _make_constraints(w, h)
    
    def objective(z):
        return -np.sum(z[2*N:])
    
    try:
        res = minimize(
            objective, z0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False}
        )
        
        if time.time() - start > time_budget:
            return None
        
        if res.success:
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            r_opt = np.maximum(r_opt, 0)
            
            # Check validity
            if _is_valid(x_opt, y_opt, r_opt, w, h):
                return x_opt, y_opt, r_opt
        return None
    except Exception:
        return None

def _polish_radii(x, y, r, w, h, time_budget, max_rounds=20):
    """Iteratively expand radii until no more improvement possible."""
    start = time.time()
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_rounds):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(N):
            # Max radius limited by walls
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            
            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            if r_max > r[i] + 1e-10:
                r[i] = max(r_max, 1e-10)
                improved = True
        
        if not improved:
            break
    
    return x, y, r

def _force_resolve(x, y, r, w, h, max_iter=500, dt=0.005):
    """Force-directed relaxation to resolve overlaps and boundary violations."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    for _ in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 100
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 100
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 100
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 100
        
        # Repulsion forces between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    d = np.sqrt(dx*dx + dy*dy)
                
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    force = overlap * 50
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d
        
        # Apply forces with damping
        x += fx * dt
        y += fy * dt
        
        # Clamp to boundaries
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    
    return x, y, r

def _hexagonal_init(w, h, n, seed=0):
    """Generate a hexagonal close-packed layout."""
    random.seed(seed)
    
    # Estimate optimal radius
    r_est = np.sqrt((w * h) / (n * 2 * np.sqrt(3)))
    
    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal offset
        offset = 0.5 * (row % 2)
        x[i] = (col + 0.5 + offset) * (w / (cols + 0.5))
        y[i] = (row + 0.5) * (h / rows)
    
    # Calculate initial radius
    min_dx = w / (cols + 0.5)
    min_dy = h / rows
    r_init = min(min_dx, min_dy) * 0.4
    r_init = min(r_init, w/2, h/2)
    
    r = np.zeros(n)
    for i in range(n):
        r[i] = min(r_init, x[i], w - x[i], y[i], h - y[i])
    
    # Add small random perturbation to break symmetry
    if seed > 0:
        noise = 0.01 * np.random.random(n)
        x = np.clip(x + (noise - 0.5) * 0.02, r, w - r)
        y = np.clip(y + (noise - 0.5) * 0.02, r, h - r)
    
    return x, y, r

def _random_init(w, h, n, seed=0):
    """Generate a random initial packing."""
    random.seed(seed)
    np.random.seed(seed)
    
    x = np.random.uniform(0, w, n)
    y = np.random.uniform(0, h, n)
    r = np.full(n, 0.02)
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Search over aspect ratios: w + h = 2, so w in [0.3, 1.7]
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.45, 0.9, 1.1, 1.3, 0.75, 1.2, 0.85, 1.15]
    
    # For each aspect ratio, try multiple initializations
    inits_per_aspect = 3
    
    for w in aspect_ratios:
        if time.time() > deadline - 40:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        
        for init_idx in range(inits_per_aspect):
            if time.time() > deadline - 20:
                break
            
            # Different initializations
            if init_idx == 0:
                x, y, r = _hexagonal_init(w, h, N, seed=init_idx)
            elif init_idx == 1:
                x, y, r = _random_init(w, h, N, seed=100 + init_idx)
            else:
                x, y, r = _hexagonal_init(w, h, N, seed=10 + init_idx)
            
            # Force resolve to get a valid starting point
            x, y, r = _force_resolve(x, y, r, w, h, max_iter=300)
            
            # Polish radii
            x, y, r = _polish_radii(x, y, r, w, h, time_budget=5)
            update_best(x, y, r, w, h)
            
            # SLSQP optimization
            remaining_time = deadline - time.time() - 10
            if remaining_time > 5:
                result = _slsqp_optimize(x, y, r, w, h, time_budget=remaining_time, maxiter=200)
                if result is not None:
                    x_opt, y_opt, r_opt = result
                    # Polish again
                    x_opt, y_opt, r_opt = _polish_radii(x_opt, y_opt, r_opt, w, h, time_budget=5)
                    update_best(x_opt, y_opt, r_opt, w, h)
    
    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x, y, r = _hexagonal_init(w, h, N, seed=0)
        x, y, r = _force_resolve(x, y, r, w, h, max_iter=200)
        x, y, r = _polish_radii(x, y, r, w, h, time_budget=2)
        best_result = (x, y, r, w, h)
    
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
