"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(positions, radii, w, h, tol=1e-7):
    for i in range(N):
        if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
            return False
        if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _compute_score(positions, radii, w, h):
    if not _is_valid(positions, radii, w, h):
        return -1.0
    return np.sum(radii)

def _grid_positions(w, h, cols, rows):
    positions = []
    dx = w / cols
    dy = h / rows
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            positions.append([x, y])
    return np.array(positions[:N])

def _hexagonal_positions(w, h, cols, rows):
    positions = []
    dx = w / cols
    dy = h / rows
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if row % 2 == 1:
                x += dx * 0.25
            positions.append([x, y])
    return np.array(positions[:N])

def _initial_radii(positions, w, h):
    radii = np.zeros(N)
    for i in range(N):
        r_max = min(positions[i, 0], w - positions[i, 0],
                    positions[i, 1], h - positions[i, 1])
        for j in range(N):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-10:
                    r_max = min(r_max, dist * 0.45)
        radii[i] = max(r_max * 0.9, 0.005)
    return radii

def _relax(positions, radii, w, h, max_iter=500, deadline=None):
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(positions)
        dradii = np.zeros(N)
        
        # Boundary forces
        for i in range(N):
            if positions[i, 0] - radii[i] < 0:
                forces[i, 0] += 10.0
            if positions[i, 0] + radii[i] > w:
                forces[i, 0] -= 10.0
            if positions[i, 1] - radii[i] < 0:
                forces[i, 1] += 10.0
            if positions[i, 1] + radii[i] > h:
                forces[i, 1] -= 10.0
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
                    dradii[i] -= overlap * 0.3
                    dradii[j] -= overlap * 0.3
                elif dist < 1e-10:
                    forces[i, 0] += np.random.randn() * 0.01
                    forces[i, 1] += np.random.randn() * 0.01
                    forces[j, 0] -= np.random.randn() * 0.01
                    forces[j, 1] -= np.random.randn() * 0.01
        
        # Try to expand radii
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-10:
                        r_max = min(r_max, (dist - radii[j]))
            r_max = max(r_max, 0.001)
            dradii[i] += (r_max - radii[i]) * 0.15
        
        # Apply updates
        lr = 0.05
        positions = positions + forces * lr
        radii = radii + dradii * lr
        
        # Clamp
        radii = np.maximum(radii, 0.001)
        for i in range(N):
            positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
    
    return positions, radii

def _optimize_with_scipy(positions, radii, w, h, deadline=None):
    n = N
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = positions[i, 0]
        x0[3*i+1] = positions[i, 1]
        x0[3*i+2] = radii[i]
    
    def objective(x):
        pos = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
        rad = np.array([x[3*i+2] for i in range(n)])
        
        penalty = 0.0
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                penalty += 1000 * (pos[i, 0] - rad[i]) ** 2
            if pos[i, 0] + rad[i] > w:
                penalty += 1000 * (pos[i, 0] + rad[i] - w) ** 2
            if pos[i, 1] - rad[i] < 0:
                penalty += 1000 * (pos[i, 1] - rad[i]) ** 2
            if pos[i, 1] + rad[i] > h:
                penalty += 1000 * (pos[i, 1] + rad[i] - h) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += 1000 * (min_dist - dist) ** 2
        
        return -np.sum(rad) + penalty
    
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0.001, min(w, h)/2))
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-9})
        
        if result.success or result.fun < 0:
            pos = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
            rad = np.array([result.x[3*i+2] for i in range(n)])
            
            if _is_valid(pos, rad, w, h):
                return pos, rad
    except:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    # Search over aspect ratios
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    
    # Initial configurations
    configs = [
        ("grid_5x5", 5, 5, _grid_positions),
        ("hex_5x5", 5, 5, _hexagonal_positions),
        ("grid_4x6", 4, 6, _grid_positions),
        ("grid_6x4", 6, 4, _grid_positions),
        ("hex_4x6", 4, 6, _hexagonal_positions),
        ("hex_6x4", 6, 4, _hexagonal_positions),
        ("grid_3x7", 3, 7, _grid_positions),
        ("grid_7x3", 7, 3, _grid_positions),
    ]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        w = 2.0 * ar
        h = 2.0 / ar
        
        for name, cols, rows, gen_func in configs:
            if time.time() > deadline - 2.0:
                break
            
            positions = gen_func(w, h, cols, rows)
            radii = _initial_radii(positions, w, h)
            
            # Relax
            positions, radii = _relax(positions, radii, w, h, max_iter=300, deadline=deadline)
            
            score = _compute_score(positions, radii, w, h)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([positions, radii])
            
            # Scipy optimization
            if time.time() < deadline - 10.0:
                opt_result = _optimize_with_scipy(positions, radii, w, h, deadline=deadline)
                if opt_result is not None:
                    pos, rad = opt_result
                    score = _compute_score(pos, rad, w, h)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([pos, rad])
    
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
