"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles, w, h, tol=1e-9):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _make_objective(w, h, lam=100.0):
    """Create a smooth objective function for circle packing."""
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        # Penalty for out of bounds (smooth)
        penalty = 0.0
        for i in range(N):
            # Left boundary
            violation = rad[i] - pos[i][0]
            if violation > 0:
                penalty += lam * violation * violation
            # Right boundary
            violation = pos[i][0] - (w - rad[i])
            if violation > 0:
                penalty += lam * violation * violation
            # Bottom boundary
            violation = rad[i] - pos[i][1]
            if violation > 0:
                penalty += lam * violation * violation
            # Top boundary
            violation = pos[i][1] - (h - rad[i])
            if violation > 0:
                penalty += lam * violation * violation
        
        # Penalty for overlap (smooth)
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist = rad[i] + rad[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    violation = min_dist - dist
                    penalty += lam * violation * violation
        
        return -np.sum(rad) + penalty
    
    return objective

def _local_optimize_slsqp(w, h, init_params, deadline, maxiter=500):
    """Local optimization using SLSQP with explicit constraints."""
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        # Penalty for out of bounds
        penalty = 0.0
        for i in range(N):
            if pos[i][0] < rad[i]:
                penalty += (rad[i] - pos[i][0]) * 500
            if pos[i][0] > w - rad[i]:
                penalty += (pos[i][0] - (w - rad[i])) * 500
            if pos[i][1] < rad[i]:
                penalty += (rad[i] - pos[i][1]) * 500
            if pos[i][1] > h - rad[i]:
                penalty += (pos[i][1] - (h - rad[i])) * 500
        
        # Penalty for overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) * 500
        
        return -np.sum(rad) + penalty
    
    # Bounds: positions within [0, w] x [0, h], radii positive
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
    for i in range(N):
        bounds.append((1e-6, min(w, h)/2))  # radius
    
    try:
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            return None, 0.0
        
        result = minimize(objective, init_params, method='SLSQP', 
                          bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False})
        
        if result.fun < 1e9:
            pos = result.x[:2*N].reshape(N, 2)
            rad = result.x[2*N:]
            
            # Clip to ensure validity
            for i in range(N):
                rad[i] = max(1e-6, min(rad[i], min(w, h)/2))
                pos[i][0] = max(rad[i], min(pos[i][0], w - rad[i]))
                pos[i][1] = max(rad[i], min(pos[i][1], h - rad[i]))
            
            circles = np.column_stack([pos, rad])
            if _validate(circles, w, h):
                return result.x, np.sum(rad)
    except Exception:
        pass
    
    return None, 0.0

def _local_optimize_nm(w, h, init_params, deadline, maxiter=2000):
    """Local optimization using Nelder-Mead."""
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        penalty = 0.0
        for i in range(N):
            if pos[i][0] < rad[i]:
                penalty += (rad[i] - pos[i][0]) * 1000
            if pos[i][0] > w - rad[i]:
                penalty += (pos[i][0] - (w - rad[i])) * 1000
            if pos[i][1] < rad[i]:
                penalty += (rad[i] - pos[i][1]) * 1000
            if pos[i][1] > h - rad[i]:
                penalty += (pos[i][1] - (h - rad[i])) * 1000
        
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) * 1000
        
        return -np.sum(rad) + penalty
    
    try:
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            return None, 0.0
        
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        
        if result.fun < 1e9:
            pos = result.x[:2*N].reshape(N, 2)
            rad = result.x[2*N:]
            
            for i in range(N):
                rad[i] = max(1e-6, min(rad[i], min(w, h)/2))
                pos[i][0] = max(rad[i], min(pos[i][0], w - rad[i]))
                pos[i][1] = max(rad[i], min(pos[i][1], h - rad[i]))
            
            circles = np.column_stack([pos, rad])
            if _validate(circles, w, h):
                return result.x, np.sum(rad)
    except Exception:
        pass
    
    return None, 0.0

def _grid_init(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            if stagger and j % 2 == 1:
                offset = 0.25 * w / nc
                x += offset
                # Clamp to stay in bounds
                x = max(0.1, min(x, w - 0.1))
            positions.append([x, y])
    
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx % 3 - 1) * 0.2 * w
        y = h / 2 + (idx // 3 % 3 - 1) * 0.2 * h
        x = max(0.1, min(x, w - 0.1))
        y = max(0.1, min(y, h - 0.1))
        positions.append([x, y])
    
    return np.array(positions[:N])

def _random_init(w, h, n_circles, rng):
    """Generate random initial positions."""
    positions = []
    for _ in range(n_circles):
        x = rng.uniform(0.1, w - 0.1)
        y = rng.uniform(0.1, h - 0.1)
        positions.append([x, y])
    return np.array(positions[:N])

def _hex_init(w, h, rng):
    """Generate hexagonal packing initial positions."""
    positions = []
    # Try to fit hexagonal pattern
    r_est = min(w, h) / 6.0
    dx = 2 * r_est * 0.866  # horizontal spacing for hex
    dy = 2 * r_est * 0.75   # vertical spacing for hex
    
    row = 0
    while len(positions) < N:
        col = 0
        while len(positions) < N:
            x = r_est + col * dx
            if row % 2 == 1:
                x += dx / 2
            y = r_est + row * dy
            
            if x < w - r_est * 0.5 and y < h - r_est * 0.5:
                positions.append([x, y])
            col += 1
        row += 1
        if row > 20:
            break
    
    # Fill remaining with random
    while len(positions) < N:
        x = rng.uniform(0.1, w - 0.1)
        y = rng.uniform(0.1, h - 0.1)
        positions.append([x, y])
    
    return np.array(positions[:N])

def _adaptive_radius_growth(w, h, init_positions, deadline, max_iter=200):
    """Adaptively grow radii while resolving collisions."""
    pos = init_positions.copy()
    # Start with small radii
    min_spacing = 1e-4
    radii = np.full(N, min_spacing)
    
    # Compute initial max possible radius based on spacing
    max_r = min(w, h) / 4.0
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Try to grow all radii equally
        growth = (max_r - radii[0]) / (max_iter - iteration)
        new_radii = radii + growth
        
        # Check for violations and shrink if needed
        valid = True
        for i in range(N):
            if new_radii[i] > w/2 or new_radii[i] > h/2:
                new_radii[i] = min(w/2, h/2) - 1e-6
                valid = False
            # Boundary check
            if pos[i][0] < new_radii[i] or pos[i][0] > w - new_radii[i]:
                valid = False
                break
            if pos[i][1] < new_radii[i] or pos[i][1] > h - new_radii[i]:
                valid = False
                break
        
        if not valid:
            # Try smaller growth
            growth *= 0.5
            new_radii = radii + growth
            if growth < 1e-6:
                break
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radii[i] + new_radii[j] - 1e-9:
                    # Overlap detected, resolve by shrinking the larger or moving
                    # Simple resolution: shrink both slightly
                    overlap = (new_radii[i] + new_radii[j]) - dist
                    if overlap > 0:
                        # Move apart
                        if dist > 1e-10:
                            nx = dx / dist
                            ny = dy / dist
                            move = overlap / 2
                            pos[i][0] += nx * move
                            pos[i][1] += ny * move
                            pos[j][0] -= nx * move
                            pos[j][1] -= ny * move
                        # Clamp positions
                        for k in [i, j]:
                            pos[k][0] = max(new_radii[k], min(pos[k][0], w - new_radii[k]))
                            pos[k][1] = max(new_radii[k], min(pos[k][1], h - new_radii[k]))
        
        # Final boundary clamp
        for i in range(N):
            pos[i][0] = max(new_radii[i], min(pos[i][0], w - new_radii[i]))
            pos[i][1] = max(new_radii[i], min(pos[i][1], h - new_radii[i]))
        
        radii = new_radii.copy()
    
    circles = np.column_stack([pos, radii])
    if _validate(circles, w, h):
        return circles, np.sum(radii)
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    rng = np.random.RandomState(42)
    
    # Try multiple aspect ratios
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    aspect_ratios = np.linspace(0.5, 2.0, 15)
    # Add some specific promising ratios
    aspect_ratios = np.unique(np.concatenate([aspect_ratios, [1.0, 1.2, 0.8, 1.5, 0.7, 1.3, 0.9, 1.1]]))
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try different initialization strategies
        inits = []
        
        # Grid layouts
        layouts = [
            (5, 4, False),
            (4, 5, False),
            (3, 7, False),
            (7, 3, False),
            (6, 4, True),
            (5, 5, True),
            (2, 11, False),
            (11, 2, False),
            (7, 3, True),
            (3, 7, True),
        ]
        
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 10.0:
                break
            positions = _grid_init(w, h, nc, nr, stagger)
            inits.append(positions)
        
        # Hexagonal
        hex_pos = _hex_init(w, h, rng)
        inits.append(hex_pos)
        
        # Random inits
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            rand_pos = _random_init(w, h, N, rng)
            inits.append(rand_pos)
        
        for positions in inits:
            if time.time() > deadline - 8.0:
                break
            
            # Initial radii: estimate based on grid spacing
            # Use a conservative estimate
            r0 = min(w, h) / 6.0 * 0.5
            radii = np.full(N, r0)
            
            init_params = np.concatenate([positions.flatten(), radii])
            
            # Try SLSQP first (better for constrained problems)
            result_params, value = _local_optimize_slsqp(w, h, init_params, deadline, maxiter=300)
            
            if result_params is not None and value > best_value:
                best_value = value
                pos = result_params[:2*N].reshape(N, 2)
                rad = result.x[2*N:]
                best_result = np.column_stack([pos, rad])
            
            # Also try Nelder-Mead from the same init
            if time.time() > deadline - 5.0:
                break
            result_params2, value2 = _local_optimize_nm(w, h, init_params, deadline, maxiter=1000)
            
            if result_params2 is not None and value2 > best_value:
                best_value = value2
                pos = result_params2[:2*N].reshape(N, 2)
                rad = result_params2[2*N:]
                best_result = np.column_stack([pos, rad])
            
            # Try adaptive radius growth
            if time.time() > deadline - 3.0:
                break
            circles_ag, value_ag = _adaptive_radius_growth(w, h, positions.copy(), deadline, max_iter=100)
            if circles_ag is not None and value_ag > best_value:
                best_value = value_ag
                best_result = circles_ag
    
    # If no valid result found, return a safe default
    if best_result is None:
        # Fallback: simple grid in a square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
