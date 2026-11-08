"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _valid(circles):
    """Check if circles are valid: inside [0,1]^2 with disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
        if r < 0:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True

def _score(circles):
    if not _valid(circles):
        return -np.inf
    return sum(c[2] for c in circles)

def _grow_and_push(positions, radii, max_iters=5000, grow_rate=0.001, push_iters=50):
    """Iteratively push circles apart and grow radii."""
    n = len(radii)
    positions = positions.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        # Check for violations and push
        violated = False
        for _ in range(push_iters):
            max_push = 0
            for i in range(n):
                # Boundary constraints
                if positions[i][0] < radii[i]:
                    positions[i][0] = radii[i]
                    violated = True
                    max_push = max(max_push, radii[i] - positions[i][0])
                if positions[i][0] > 1 - radii[i]:
                    positions[i][0] = 1 - radii[i]
                    violated = True
                    max_push = max(max_push, positions[i][0] - (1 - radii[i]))
                if positions[i][1] < radii[i]:
                    positions[i][1] = radii[i]
                    violated = True
                    max_push = max(max_push, radii[i] - positions[i][1])
                if positions[i][1] > 1 - radii[i]:
                    positions[i][1] = 1 - radii[i]
                    violated = True
                    max_push = max(max_push, positions[i][1] - (1 - radii[i]))
            
            # Pairwise overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist - 1e-12:
                        if dist < 1e-12:
                            # Random direction
                            angle = np.random.uniform(0, 2*np.pi)
                            dx, dy = np.cos(angle), np.sin(angle)
                            dist = 1.0
                        overlap = (min_dist - dist) / 2
                        nx, ny = dx/dist, dy/dist
                        positions[i][0] += overlap * nx
                        positions[i][1] += overlap * ny
                        positions[j][0] -= overlap * nx
                        positions[j][1] -= overlap * ny
                        violated = True
                        max_push = max(max_push, overlap)
            
            if not violated:
                break
        
        if violated:
            continue
        
        # Grow radii
        radii[:] *= (1 + grow_rate)
        
        # Check if still valid after growth
        valid = True
        for i in range(n):
            if positions[i][0] < radii[i] or positions[i][0] > 1 - radii[i]:
                valid = False
                break
            if positions[i][1] < radii[i] or positions[i][1] > 1 - radii[i]:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
        if not valid:
            # Shrink back
            radii[:] /= (1 + grow_rate)
            break
    
    return positions, radii

def _scipy_optimize(circles, max_iters=1000, timeout=30):
    """Use scipy to optimize positions and radii to maximize sum of radii."""
    n = len(circles)
    # Variables: x1,y1,r1, x2,y2,r2, ..., xn,yn,rn
    x0 = circles.flatten().copy()
    
    def objective(v):
        # Maximize sum of radii = minimize negative sum of radii
        circles_v = v.reshape(n, 3)
        # Penalty for violations
        penalty = 0
        for i in range(n):
            x, y, r = circles_v[i]
            if r < 0:
                penalty += (r + 0.1)**2
            if x < r:
                penalty += (r - x)**2
            if x > 1 - r:
                penalty += (x - (1 - r))**2
            if y < r:
                penalty += (r - y)**2
            if y > 1 - r:
                penalty += (y - (1 - r))**2
        for i in range(n):
            for j in range(i+1, n):
                dx = circles_v[i][0] - circles_v[j][0]
                dy = circles_v[i][1] - circles_v[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles_v[i][2] + circles_v[j][2]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        return -np.sum(circles_v[:, 2]) + penalty
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0.001, 0.5))  # r
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iters, 'maxfun': 10000})
    
    if result.success or result.fun < objective(x0):
        new_circles = result.x.reshape(n, 3)
        if _valid(new_circles):
            return new_circles
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    configs = []
    
    # Grid 5x5 (25 circles) + 1 extra
    side = 5
    for i in range(side):
        for j in range(side):
            configs.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005])
    # Add 26th circle in a corner-ish spot
    configs.append([0.1, 0.1, 0.05])
    
    # Grid 6x5
    for i in range(6):
        for j in range(5):
            configs.append([(i + 0.5) / 6, (j + 0.5) / 5, min(0.5/6, 0.5/5) - 0.005])
    
    # Hexagonal-ish packing
    for row in range(5):
        offset = 0.5/6 if row % 2 == 0 else 0
        for col in range(6):
            x = (col + 0.5) / 6 + offset
            y = (row + 0.5) / 5
            if x <= 1:
                configs.append([x, y, min(0.5/6, 0.5/5) - 0.005])
    
    for config in configs[:5]:  # Limit number of configs
        if time.time() > deadline:
            break
        
        circles = np.array(config[:N])
        if len(circles) < N:
            continue
        
        # Run grow_and_push
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Initial: make sure valid
        if not _valid(circles):
            continue
        
        positions, radii = _grow_and_push(positions, radii, max_iters=3000, grow_rate=0.002)
        
        result_circles = np.column_stack([positions, radii])
        if _valid(result_circles):
            s = _score(result_circles)
            if s > best_score:
                best_score = s
                best_circles = result_circles
        
        # Try scipy optimization from this result
        if time.time() < deadline - 20:
            opt_result = _scipy_optimize(result_circles, max_iters=500, timeout=20)
            if opt_result is not None:
                s = _score(opt_result)
                if s > best_score:
                    best_score = s
                    best_circles = opt_result
        
        # Try scipy from original config
        if time.time() < deadline - 20:
            opt_result = _scipy_optimize(circles, max_iters=500, timeout=20)
            if opt_result is not None:
                s = _score(opt_result)
                if s > best_score:
                    best_score = s
                    best_circles = opt_result
    
    # Final: if we have a good result, do one more scipy pass
    if best_circles is not None and time.time() < deadline - 10:
        opt_result = _scipy_optimize(best_circles, max_iters=1000, timeout=15)
        if opt_result is not None:
            s = _score(opt_result)
            if s > best_score:
                best_score = s
                best_circles = opt_result
    
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
