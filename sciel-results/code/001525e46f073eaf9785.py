import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _check_valid(x, y, r, W, H, tol=1e-7):
    """Check validity: circles within bounds, non-overlapping."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _slsqp_optimize(x0, y0, r0, W, H, timeout=8.0):
    """SLSQP optimization: maximize sum(r) subject to bounds and non-overlap."""
    start = time.time()
    n = len(x0)
    
    def objective(v):
        if time.time() - start > timeout:
            return 1e10
        xi, yi, ri = v[:n], v[n:2*n], v[2*n:3*n]
        return -np.sum(ri)
    
    # Constraints: non-overlap (dist >= ri + rj)
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(i, j):
                def con(v):
                    xi, yi, ri = v[:n], v[n:2*n], v[2*n:3*n]
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - ri[i] - ri[j]
                con.name = f"overlap_{i}_{j}"
                return con
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))  # x_i
    for i in range(n):
        bounds.append((1e-6, H - 1e-6))  # y_i
    for i in range(n):
        bounds.append((1e-6, min(W, H) / 2 - 1e-6))  # r_i
    
    v0 = np.concatenate([x0, y0, r0])
    
    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 150, 'ftol': 1e-9, 'disp': False})
        if time.time() - start > timeout:
            return None
        if res.fun < -np.sum(r0) - 1e-8:  # improved
            xi, yi, ri = res.x[:n], res.x[n:2*n], res.x[2*n:3*n]
            if _check_valid(xi, yi, ri, W, H):
                return xi, yi, ri
    except Exception:
        pass
    return None

def _lbfgs_polish(x0, y0, r0, W, H, timeout=6.0):
    """L-BFGS-B with penalty for constraints."""
    start = time.time()
    n = len(x0)
    
    def objective(v):
        if time.time() - start > timeout:
            return 1e10
        xi, yi, ri = v[:n], v[n:2*n], v[2*n:3*n]
        obj = -np.sum(ri)
        # Penalty for violations
        pen = 0.0
        # Boundary
        pen += np.sum(np.maximum(0, ri - xi)**2)
        pen += np.sum(np.maximum(0, xi - (W - ri))**2)
        pen += np.sum(np.maximum(0, ri - yi)**2)
        pen += np.sum(np.maximum(0, yi - (H - ri))**2)
        # Overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                dist = np.sqrt(dx*dx + dy*dy)
                viol = max(0, ri[i] + ri[j] - dist)
                pen += viol**2
        return obj + 50.0 * pen
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
    for i in range(n):
        bounds.append((1e-6, H - 1e-6))
    for i in range(n):
        bounds.append((1e-6, min(W, H) / 2 - 1e-6))
    
    v0 = np.concatenate([x0, y0, r0])
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'disp': False})
        if time.time() - start > timeout:
            return None
        xi, yi, ri = res.x[:n], res.x[n:2*n], res.x[2*n:3*n]
        if _check_valid(xi, yi, ri, W, H):
            return xi, yi, ri
    except Exception:
        pass
    return None

def _physics_sim(x0, y0, r0, W, H, max_steps=5000, start_time=None):
    """Physics-based simulation with repulsive forces and radius growth."""
    if start_time is None:
        start_time = time.time()
    
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    n = len(x)
    
    dt = 0.002
    damping = 0.92
    growth_rate = 0.0003
    
    # Velocity
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    for step in range(max_steps):
        if _elapsed(start_time) > DEADLINE - 10:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist + 0.05:  # interaction range
                    if dist < 1e-10:
                        dx = 0.001
                        dy = 0.001
                        dist = np.sqrt(dx*dx + dy*dy)
                    # Repulsive force (spring-like)
                    overlap = min_dist - dist
                    if overlap > 0:
                        force = 50.0 * overlap
                        fx[i] += dx / dist * force
                        fy[i] += dy / dist * force
                        fx[j] -= dx / dist * force
                        fy[j] -= dy / dist * force
                    else:
                        # Mild attraction to prevent drift (LJ-like)
                        force = -0.1 * (min_dist - dist)
                        fx[i] += dx / dist * force
                        fy[i] += dy / dist * force
                        fx[j] -= dx / dist * force
                        fy[j] -= dy / dist * force
        
        # Wall forces (soft)
        wall_k = 100.0
        for i in range(n):
            # Left wall
            if x[i] < r[i]:
                fx[i] += wall_k * (r[i] - x[i])
            # Right wall
            if x[i] > W - r[i]:
                fx[i] -= wall_k * (x[i] - (W - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += wall_k * (r[i] - y[i])
            # Top wall
            if y[i] > H - r[i]:
                fy[i] -= wall_k * (y[i] - (H - r[i]))
        
        # Update velocities and positions
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        x += vx * dt
        y += vy * dt
        
        # Hard clamp to boundaries
        for i in range(n):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
        
        # Grow radii
        r += growth_rate
        # Cap radius by distance to walls
        for i in range(n):
            max_r = min(x[i], W - x[i], y[i], H - y[i])
            if r[i] > max_r:
                r[i] = max_r
        
        # Check convergence
        if step > 100 and np.max(np.abs(vx)) < 1e-6 and np.max(np.abs(vy)) < 1e-6:
            # Try to grow more
            if np.all(r < min(W, H) / 2 - 1e-6):
                r += 0.001
            else:
                break
        
        # Periodically check validity
        if step % 500 == 0 and _check_valid(x, y, r, W, H):
            pass  # valid, continue
    
    return x, y, r

def _init_hex(W, H, rng):
    """Hexagonal packing initialization."""
    x = []
    y = []
    # Estimate radius
    rows = 5
    r_est = min(W / (rows * 1.8 + 1), H / (rows * 1.6 + 1)) * 0.85
    r_est = min(r_est, 0.12)
    
    row_spacing = r_est * np.sqrt(3) * 1.05
    col_spacing = 2 * r_est * 1.05
    row = 0
    while len(x) < N:
        cols_in_row = int(np.floor((W - 2*r_est) / col_spacing)) + 1
        offset = col_spacing / 2 if row % 2 == 1 else 0
        for c in range(cols_in_row):
            if len(x) >= N:
                break
            cx = r_est + offset + c * col_spacing
            cy = r_est + row * row_spacing
            if cx + r_est <= W and cy + r_est <= H:
                x.append(cx)
                y.append(cy)
        row += 1
    
    if len(x) < N:
        return None
    
    return np.array(x[:N]), np.array(y[:N]), np.full(N, r_est)

def _init_grid(W, H, rng, m, k):
    """Grid initialization with jitter."""
    if m * k < N:
        return None
    xs = []
    ys = []
    for i in range(m):
        for j in range(k):
            if len(xs) >= N:
                break
            cx = (i + 0.5) * W / m
            cy = (j + 0.5) * H / k
            # Add small jitter
            cx += rng.uniform(-0.01, 0.01)
            cy += rng.uniform(-0.01, 0.01)
            xs.append(cx)
            ys.append(cy)
    
    if len(xs) < N:
        return None
    
    r_est = min(W / (2*m) * 0.9, H / (2*k) * 0.9)
    return np.array(xs[:N]), np.array(ys[:N]), np.full(N, r_est)

def _init_ring(W, H, rng):
    """Ring-based initialization: one center + rings around."""
    cx, cy = W/2, H/2
    x = [cx]
    y = [cy]
    
    # Ring 1: 6 circles
    r1 = min(W, H) * 0.15
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        px = cx + r1 * np.cos(angle)
        py = cy + r1 * np.sin(angle)
        if 0 < px < W and 0 < py < H:
            x.append(px)
            y.append(py)
    
    # Ring 2: more circles
    r2 = min(W, H) * 0.28
    n_ring2 = N - len(x)
    for i in range(n_ring2):
        angle = 2 * np.pi * i / n_ring2
        px = cx + r2 * np.cos(angle)
        py = cy + r2 * np.sin(angle)
        if 0 < px < W and 0 < py < H:
            x.append(px)
            y.append(py)
        else:
            # Place along edges
            px = rng.uniform(0.1, W - 0.1)
            py = rng.uniform(0.1, H - 0.1)
            x.append(px)
            y.append(py)
    
    if len(x) < N:
        return None
    
    r_est = min(W, H) * 0.08
    return np.array(x[:N]), np.array(y[:N]), np.full(N, r_est)

def _init_random(W, H, rng, seed_offset=0):
    """Random initialization with rejection sampling."""
    x = []
    y = []
    r_est = min(W, H) * 0.08
    
    attempts = 0
    while len(x) < N and attempts < 5000:
        px = rng.uniform(r_est, W - r_est)
        py = rng.uniform(r_est, H - r_est)
        # Check no overlap with existing
        ok = True
        for i in range(len(x)):
            dx = px - x[i]
            dy = py - y[i]
            if dx*dx + dy*dy < (2*r_est*1.1)**2:
                ok = False
                break
        if ok:
            x.append(px)
            y.append(py)
        attempts += 1
    
    if len(x) < N:
        return None
    
    return np.array(x[:N]), np.array(y[:N]), np.full(N, r_est)

def construct():
    start_time = time.time()
    best_value = -np.inf
    best_result = None
    
    rng = np.random.RandomState(42)
    
    # Aspect ratios to try
    aspects = [1.0, 0.85, 1.18, 0.75, 1.33, 0.9, 1.1, 0.8, 1.25, 0.7, 1.43]
    
    # For each aspect, generate initializations and optimize
    for aspect in aspects:
        if _elapsed(start_time) > DEADLINE - 15:
            break
        
        W = 2.0 * aspect / (1 + aspect)
        H = 2.0 - W
        
        # Generate multiple initializations
        inits = []
        
        # Hex
        hex_init = _init_hex(W, H, rng)
        if hex_init is not None:
            inits.append(hex_init)
        
        # Grids
        for m, k in [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (5, 4), (4, 5), (3, 8), (8, 3)]:
            g = _init_grid(W, H, rng, m, k)
            if g is not None:
                inits.append(g)
        
        # Ring
        ring_init = _init_ring(W, H, rng)
        if ring_init is not None:
            inits.append(ring_init)
        
        # Random
        for _ in range(2):
            r_init = _init_random(W, H, rng)
            if r_init is not None:
                inits.append(r_init)
        
        # Deduplicate similar inits (by radius estimate)
        if not inits:
            continue
        
        for x0, y0, r0 in inits:
            if _elapsed(start_time) > DEADLINE - 10:
                break
            
            # Physics simulation to settle
            x_sim, y_sim, r_sim = _physics_sim(x0, y0, r0, W, H, 
                                                max_steps=3000, start_time=start_time)
            
            # Validate after physics
            if _check_valid(x_sim, y_sim, r_sim, W, H):
                val = np.sum(r_sim)
                if val > best_value:
                    best_value = val
                    best_result = np.column_stack([x_sim, y_sim, r_sim])
            
            # SLSQP polish
            if _elapsed(start_time) < DEADLINE - 12:
                polished = _slsqp_optimize(x_sim, y_sim, r_sim, W, H, timeout=6.0)
                if polished is not None:
                    px, py, pr = polished
                    pval = np.sum(pr)
                    if pval > best_value:
                        best_value = pval
                        best_result = np.column_stack([px, py, pr])
            
            # L-BFGS polish
            if _elapsed(start_time) < DEADLINE - 8:
                lbfgs_res = _lbfgs_polish(x_sim, y_sim, r_sim, W, H, timeout=4.0)
                if lbfgs_res is not None:
                    lx, ly, lr = lbfgs_res
                    lval = np.sum(lr)
                    if lval > best_value:
                        best_value = lval
                        best_result = np.column_stack([lx, ly, lr])
    
    # Final fallback
    if best_result is None:
        W, H = 1.0, 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
        best_result = np.array(cells[:N])
    
    return best_result
