"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
_start_time = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _check_feasibility(circles):
    """Check if circles are valid (inside unit square, no overlaps)."""
    if len(circles) != N:
        return False
    for x, y, r in circles:
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2 - 1e-9:
                return False
    return True

def _sum_radii(circles):
    return sum(c[2] for c in circles)

def _initial_grid(n=N):
    """Create initial grid of circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    # Fill remaining if needed
    while len(cells) < n:
        cells.append([0.5, 0.5, 0.01])
    return np.array(cells[:n])

def _optimize_from_initial(initial, max_iter=200, tol=1e-6):
    """Optimize circle positions and radii using SLSQP with hard constraints."""
    n = len(initial)
    x0 = initial.flatten().copy()
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    # Bounds: 0 <= x <= 1, 0 <= y <= 1, 0 <= r <= 0.5
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    # Constraints:
    # 1. x - r >= 0 (left boundary)
    # 2. 1 - x - r >= 0 (right boundary)
    # 3. y - r >= 0 (bottom boundary)
    # 4. 1 - y - r >= 0 (top boundary)
    # 5. dist(i,j) >= r_i + r_j for all i < j
    
    def boundary_constraints(x):
        cons = []
        for i in range(n):
            xi = x[3*i]
            yi = x[3*i + 1]
            ri = x[3*i + 2]
            cons.append(xi - ri)
            cons.append(1 - xi - ri)
            cons.append(yi - ri)
            cons.append(1 - yi - ri)
        return cons
    
    def overlap_constraints(x):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                xi = x[3*i]
                yi = x[3*i + 1]
                ri = x[3*i + 2]
                xj = x[3*j]
                yj = x[3*j + 1]
                rj = x[3*j + 2]
                # dist >= ri + rj  =>  dist^2 >= (ri + rj)^2
                # Use: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                cons.append((xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2)
        return cons
    
    def objective(x):
        # Maximize sum of radii => minimize negative sum
        radii = x[2::3]
        return -np.sum(radii)
    
    # Combine constraints
    def all_constraints(x):
        return boundary_constraints(x) + overlap_constraints(x)
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=all_constraints,
        options={'maxiter': max_iter, 'ftol': tol}
    )
    
    if result.success or result.fun < -_sum_radii(initial):
        return result.x.reshape(-1, 3)
    return initial

def construct():
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return _initial_grid()
    
    best_circles = _initial_grid()
    best_score = _sum_radii(best_circles)
    
    # Try multiple initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        if restart == 0:
            initial = _initial_grid()
        elif restart == 1:
            # Slightly perturbed grid
            initial = _initial_grid() + np.random.randn(N, 3) * 0.01
            # Clip to valid range
            initial[:, 0] = np.clip(initial[:, 0], 0.01, 0.99)
            initial[:, 1] = np.clip(initial[:, 1], 0.01, 0.99)
            initial[:, 2] = np.clip(initial[:, 2], 0.01, 0.4)
        elif restart == 2:
            # Different grid arrangement
            side = 6
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        r = 0.5 / side - 0.005
                        cells.append([x, y, r])
            initial = np.array(cells[:N])
        elif restart == 3:
            # Random initial
            initial = np.random.rand(N, 3)
            initial[:, 0] = initial[:, 0] * 0.8 + 0.1
            initial[:, 1] = initial[:, 1] * 0.8 + 0.1
            initial[:, 2] = initial[:, 2] * 0.3 + 0.02
        else:
            # Another perturbed grid
            initial = _initial_grid() + np.random.randn(N, 3) * 0.02
            initial[:, 0] = np.clip(initial[:, 0], 0.01, 0.99)
            initial[:, 1] = np.clip(initial[:, 1], 0.01, 0.99)
            initial[:, 2] = np.clip(initial[:, 2], 0.01, 0.4)
        
        # Ensure initial is feasible (clip radii to avoid obvious overlaps)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((initial[i, 0] - initial[j, 0])**2 + (initial[i, 1] - initial[j, 1])**2)
                max_r = dist / 2 - 0.001
                if initial[i, 2] > max_r:
                    initial[i, 2] = max_r
                if initial[j, 2] > max_r:
                    initial[j, 2] = max_r
            # Also check boundary
            initial[i, 2] = min(initial[i, 2], initial[i, 0], 1 - initial[i, 0], initial[i, 1], 1 - initial[i, 1])
            initial[i, 2] = max(initial[i, 2], 0.001)
        
        result = _optimize_from_initial(initial, max_iter=300)
        
        if _check_feasibility(result):
            score = _sum_radii(result)
            if score > best_score:
                best_score = score
                best_circles = result
    
    return best_circles
# EVOLVE-BLOCK-END
