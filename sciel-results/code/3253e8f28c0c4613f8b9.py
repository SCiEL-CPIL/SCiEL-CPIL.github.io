"""Improved circle packing for n=32 in unit square."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def _validity_check(arr):
    """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return False
    if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
        return False
    if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _sum_radii(arr):
    return np.sum(arr[:, 2])

def _objective(params):
    """Objective to minimize: negative sum of radii plus penalty for violations."""
    n = N
    # Unpack: positions (2n) and radii (n)
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    x, y = pos[:, 0], pos[:, 1]
    
    penalty = 0.0
    
    # Boundary penalties
    for i in range(n):
        if x[i] < radii[i]:
            penalty += (radii[i] - x[i])**2
        if x[i] > 1 - radii[i]:
            penalty += (x[i] - (1 - radii[i]))**2
        if y[i] < radii[i]:
            penalty += (radii[i] - y[i])**2
        if y[i] > 1 - radii[i]:
            penalty += (y[i] - (1 - radii[i]))**2
    
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return -np.sum(radii) + 100.0 * penalty

def _hexagonal_init():
    """Create a hexagonal lattice initial configuration."""
    # Use a hexagonal packing pattern
    points = []
    # Try different scales
    for scale in [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06]:
        # Hexagonal grid
        row = 0
        while len(points) < N + 10:
            x_offset = 0 if row % 2 == 0 else scale * 0.866
            col = 0
            while True:
                x = x_offset + col * scale * 2 * 0.866
                y = row * scale * 0.866
                if x > 1 or y > 1:
                    break
                points.append((x, y))
                col += 1
            row += 1
        # Select N points that fit well
        valid_points = [(x, y) for x, y in points if 0 <= x <= 1 and 0 <= y <= 1]
        if len(valid_points) >= N:
            # Use first N
            pts = valid_points[:N]
            # Estimate radius
            min_dist = float('inf')
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = np.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
                    min_dist = min(min_dist, d)
            # Also consider boundary
            for px, py in pts:
                min_dist = min(min_dist, px, py, 1-px, 1-py)
            r = min_dist / 2 - 0.005
            if r > 0.01:
                return [(px, py, r) for px, py in pts]
    return None

def _phyllotaxis_init():
    """Create a phyllotaxis (sunflower) pattern initial configuration."""
    points = []
    for i in range(N):
        angle = i * 2.399963229728653  # golden angle
        r = 0.5 * np.sqrt((i + 0.5) / N)
        x = 0.5 + r * np.cos(angle)
        y = 0.5 + r * np.sin(angle)
        points.append((x, y))
    
    # Estimate radius
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            min_dist = min(min_dist, d)
    for px, py in points:
        min_dist = min(min_dist, px, py, 1-px, 1-py)
    r = min_dist / 2 - 0.005
    if r > 0.01:
        return [(px, py, r) for px, py in points]
    return None

def _random_init(seed):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        points.append((x, y))
    
    # Estimate radius
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            min_dist = min(min_dist, d)
    for px, py in points:
        min_dist = min(min_dist, px, py, 1-px, 1-py)
    r = min_dist / 2 - 0.005
    if r > 0.01:
        return [(px, py, r) for px, py in points]
    return None

def _physics_push_apart(points, iterations=1000, k=0.001):
    """Use physics-based push-apart to resolve overlaps."""
    n = len(points)
    pts = np.array([(p[0], p[1]) for p in points], dtype=float)
    r = points[0][2]  # assume equal radii for now
    
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = 2 * r
                if dist < min_dist and dist > 1e-10:
                    force = k * (min_dist - dist) / dist
                    forces[i,0] += dx * force
                    forces[i,1] += dy * force
                    forces[j,0] -= dx * force
                    forces[j,1] -= dy * force
        
        # Boundary forces
        for i in range(n):
            if pts[i,0] < r:
                forces[i,0] += k * (r - pts[i,0])
            if pts[i,0] > 1-r:
                forces[i,0] -= k * (pts[i,0] - (1-r))
            if pts[i,1] < r:
                forces[i,1] += k * (r - pts[i,1])
            if pts[i,1] > 1-r:
                forces[i,1] -= k * (pts[i,1] - (1-r))
        
        pts += forces
        # Clamp to boundary
        pts[:,0] = np.clip(pts[:,0], r, 1-r)
        pts[:,1] = np.clip(pts[:,1], r, 1-r)
    
    return [(pts[i,0], pts[i,1], r) for i in range(n)]

def _optimize_from_init(init_points, deadline):
    """Run optimization from a given initial configuration."""
    n = N
    # Build initial params
    x = np.array([p[0] for p in init_points])
    y = np.array([p[1] for p in init_points])
    r = np.array([p[2] for p in init_points])
    
    # Ensure minimum radius
    r = np.maximum(r, 0.01)
    
    x0 = np.concatenate([np.column_stack([x, y]).flatten(), r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(n):
        bounds.append((0.01, 0.49))  # radius
    
    start_time = time.time()
    result = None
    
    # Try multiple optimization strategies
    strategies = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}),
        ('SLSQP', {'maxiter': 300, 'ftol': 1e-12}),
        ('COBYLA', {'maxiter': 1000, 'rhobnd': 1e-6}),
    ]
    
    best_obj = float('inf')
    best_x = x0.copy()
    
    for method, opts in strategies:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_objective, x0, method=method, bounds=bounds, options=opts)
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Extract result
    pos = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Clamp to valid range
    radii = np.clip(radii, 0.005, 0.49)
    pos[:,0] = np.clip(pos[:,0], radii, 1-radii)
    pos[:,1] = np.clip(pos[:,1], radii, 1-radii)
    
    result = np.column_stack([pos[:,0], pos[:,1], radii])
    
    # Final validity check and fix
    if not _validity_check(result):
        # Try to fix by reducing radii
        for _ in range(50):
            radii *= 0.95
            pos[:,0] = np.clip(pos[:,0], radii, 1-radii)
            pos[:,1] = np.clip(pos[:,1], radii, 1-radii)
            result = np.column_stack([pos[:,0], pos[:,1], radii])
            if _validity_check(result):
                break
    
    return result

def _tight_polish(arr, deadline):
    """High-precision polish pass."""
    n = N
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    x0 = np.concatenate([np.column_stack([x, y]).flatten(), r])
    
    bounds = []
    for i in range(n):
        bounds.append((r[i]*0.5, 1 - r[i]*0.5))  # x
        bounds.append((r[i]*0.5, 1 - r[i]*0.5))  # y
    for i in range(n):
        bounds.append((r[i]*0.8, r[i]*1.2))  # radius
    
    start_time = time.time()
    
    # Use tighter tolerance
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < _objective(x0):
            pos = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            radii = np.clip(radii, 0.005, 0.49)
            pos[:,0] = np.clip(pos[:,0], radii, 1-radii)
            pos[:,1] = np.clip(pos[:,1], radii, 1-radii)
            result = np.column_stack([pos[:,0], pos[:,1], radii])
            if _validity_check(result):
                return result
    except Exception:
        pass
    
    return arr

def construct():
    deadline = time.time() + 120  # Leave margin before 150s kill
    
    # Generate multiple initializations
    inits = []
    
    # 1. Hexagonal
    hex_init = _hexagonal_init()
    if hex_init:
        inits.append(hex_init)
    
    # 2. Phyllotaxis
    phy_init = _phyllotaxis_init()
    if phy_init:
        inits.append(phy_init)
    
    # 3. Grid (original)
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                  for i in range(side) for j in range(side)]
    inits.append(grid_cells[:N])
    
    # 4. Random initializations
    for seed in range(10):
        rand_init = _random_init(seed)
        if rand_init:
            inits.append(rand_init)
    
    # Apply physics push-apart to each init
    processed_inits = []
    for init in inits:
        if time.time() > deadline - 30:
            break
        try:
            pushed = _physics_push_apart(init, iterations=500)
            processed_inits.append(pushed)
        except Exception:
            processed_inits.append(init)
    
    # Optimize from each initialization
    best_result = None
    best_sum = -float('inf')
    
    for i, init in enumerate(processed_inits):
        if time.time() > deadline - 10:
            break
        
        result = _optimize_from_init(init, deadline)
        if result is not None and _validity_check(result):
            s = _sum_radii(result)
            if s > best_sum:
                best_sum = s
                best_result = result
    
    # Tight polish on best result
    if best_result is not None and time.time() < deadline - 5:
        polished = _tight_polish(best_result, deadline)
        if _validity_check(polished) and _sum_radii(polished) > best_sum:
            best_result = polished
    
    # Fallback to grid if nothing worked
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                      for i in range(side) for j in range(side)]
        best_result = np.array(grid_cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
