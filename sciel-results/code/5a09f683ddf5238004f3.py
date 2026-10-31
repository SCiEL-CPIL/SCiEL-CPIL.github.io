"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _valid(arr):
    """Check if configuration is valid: disjoint interiors, inside unit square."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def _score(arr):
    """Sum of radii."""
    return float(np.sum(arr[:, 2]))

def _make_initial_grid():
    """Create a grid of equal circles as initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([cx, cy, r])
    return np.array(cells)

def _greedy_best_fit():
    """Greedy best-fit initialization: place circles one by one at the location
    that maximizes the minimum distance to existing circles and boundaries."""
    arr = np.zeros((N, 3))
    # Start with a reasonable radius estimate
    # For 26 circles in unit square, average area per circle ~ 1/26
    # radius ~ sqrt(1/(4*pi*26)) ~ 0.055
    base_r = np.sqrt(1.0 / (4 * np.pi * N)) * 0.9
    
    for i in range(N):
        # Try many candidate positions
        best_pos = None
        best_min_dist = -1
        # Use a grid of candidate positions
        for gx in np.linspace(0.05, 0.95, 20):
            for gy in np.linspace(0.05, 0.95, 20):
                # Distance to boundaries
                dist_bound = min(gx, 1 - gx, gy, 1 - gy)
                # Distance to existing circles
                min_dist_to_circles = dist_bound
                for j in range(i):
                    d = np.hypot(gx - arr[j, 0], gy - arr[j, 1])
                    # We want the gap between circles to be large
                    # The effective distance is d - r_j - r_i (but r_i not set yet)
                    # Use a proxy: distance to edge of existing circle
                    gap = d - arr[j, 2]
                    if gap < min_dist_to_circles:
                        min_dist_to_circles = gap
                if min_dist_to_circles > best_min_dist:
                    best_min_dist = min_dist_to_circles
                    best_pos = (gx, gy)
        
        if best_pos is None:
            best_pos = (0.5, 0.5)
        
        arr[i, 0] = best_pos[0]
        arr[i, 1] = best_pos[1]
        # Set radius conservatively
        arr[i, 2] = min(base_r, best_min_dist * 0.95)
    
    return arr

def _optimize_slsqp(arr0, max_iter=200, tol=1e-8):
    """Optimize using SLSQP with hard constraints."""
    n = arr0.shape[0]
    x0 = arr0.flatten()
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    def objective(v):
        # Minimize negative sum of radii
        r = v[2::3]
        return -np.sum(r)
    
    # Constraints:
    # 1. Boundary: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    # 2. Pairwise: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #    Equivalent: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    constraints = []
    
    # Boundary constraints (linear)
    for i in range(n):
        xi = 3 * i
        yi = 3 * i + 1
        ri = 3 * i + 2
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[xi] - v[ri], 'jac': lambda v, xi=xi, ri=ri: _jac_boundary(v, xi, ri, 'left')})
        # x_i + r_i <= 1 => 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: 1 - v[xi] - v[ri], 'jac': lambda v, xi=xi, ri=ri: _jac_boundary(v, xi, ri, 'right')})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[yi] - v[ri], 'jac': lambda v, yi=yi, ri=ri: _jac_boundary(v, yi, ri, 'left')})
        # y_i + r_i <= 1 => 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: 1 - v[yi] - v[ri], 'jac': lambda v, yi=yi, ri=ri: _jac_boundary(v, yi, ri, 'right')})
    
    # Pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
            xj, yj, rj = 3 * j, 3 * j + 1, 3 * j + 2
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                    (v[xi] - v[xj])**2 + (v[yi] - v[yj])**2 - (v[ri] + v[rj])**2,
                'jac': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                    _jac_pair(v, xi, yi, ri, xj, yj, rj)
            })
    
    # Bounds: r_i >= 0, x_i, y_i in [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((0, 1))  # r_i
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
    
    if result.success:
        return result.x.reshape(n, 3)
    else:
        # Even if not fully successful, use the result if it's better
        return result.x.reshape(n, 3)

def _jac_boundary(v, idx_pos, idx_r, side):
    """Jacobian for boundary constraints."""
    jac = np.zeros_like(v)
    if side == 'left':
        # constraint: v[idx_pos] - v[idx_r]
        jac[idx_pos] = 1
        jac[idx_r] = -1
    else:
        # constraint: 1 - v[idx_pos] - v[idx_r]
        jac[idx_pos] = -1
        jac[idx_r] = -1
    return jac

def _jac_pair(v, xi, yi, ri, xj, yj, rj):
    """Jacobian for pairwise distance constraint:
    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
    """
    jac = np.zeros_like(v)
    dx = v[xi] - v[xj]
    dy = v[yi] - v[yj]
    r_sum = v[ri] + v[rj]
    
    jac[xi] = 2 * dx
    jac[yi] = 2 * dy
    jac[xj] = -2 * dx
    jac[yj] = -2 * dy
    jac[ri] = -2 * r_sum
    jac[rj] = -2 * r_sum
    return jac

def _penalty_optimize(arr0, max_iter=500, mu=100.0, tol=1e-8):
    """Optimize using a penalty method as backup."""
    n = arr0.shape[0]
    x0 = arr0.flatten()
    
    def objective(v):
        r = v[2::3]
        score = np.sum(r)
        # Penalty for violations
        penalty = 0
        # Boundary violations
        for i in range(n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            penalty += max(0, -v[xi] + v[ri])**2
            penalty += max(0, v[xi] + v[ri] - 1)**2
            penalty += max(0, -v[yi] + v[ri])**2
            penalty += max(0, v[yi] + v[ri] - 1)**2
        # Pairwise violations
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = 3*i, 3*i+1, 3*i+2
                xj, yj, rj = 3*j, 3*j+1, 3*j+2
                dist_sq = (v[xi]-v[xj])**2 + (v[yi]-v[yj])**2
                r_sum = v[ri] + v[rj]
                if dist_sq < r_sum**2:
                    penalty += (r_sum**2 - dist_sq)**2
        return -score + mu * penalty
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 1))
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': tol})
    return result.x.reshape(n, 3)

def _final_polish(arr, time_budget=30):
    """High-precision polish pass with tighter tolerances."""
    start = time.time()
    current = arr.copy()
    current_score = _score(current)
    
    for iteration in range(50):
        if time.time() - start > time_budget:
            break
        result = _optimize_slsqp(current, max_iter=100, tol=1e-10)
        if _valid(result):
            new_score = _score(result)
            if new_score > current_score:
                current = result
                current_score = new_score
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    # Generate multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(_make_initial_grid())
    
    # Greedy best-fit
    inits.append(_greedy_best_fit())
    
    # Random perturbations of grid
    rng = np.random.RandomState(42)
    for _ in range(3):
        grid = _make_initial_grid()
        # Perturb positions slightly
        perturbed = grid.copy()
        perturbed[:, 0] += rng.uniform(-0.02, 0.02, N)
        perturbed[:, 1] += rng.uniform(-0.02, 0.02, N)
        # Clip to [0,1]
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        inits.append(perturbed)
    
    best_result = None
    best_score = -1
    
    for init in inits:
        if time.time() > deadline - 30:  # Keep 30s for polish
            break
        
        # Try SLSQP optimization
        try:
            result = _optimize_slsqp(init, max_iter=300, tol=1e-8)
            if _valid(result):
                score = _score(result)
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
        
        # Try penalty method as backup
        if time.time() > deadline - 30:
            break
        try:
            result = _penalty_optimize(init, max_iter=300, mu=50.0, tol=1e-8)
            if _valid(result):
                score = _score(result)
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
    
    # If no valid result yet, use the grid
    if best_result is None:
        best_result = _make_initial_grid()
        best_score = _score(best_result)
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            polished = _final_polish(best_result, time_budget=min(25, deadline - time.time() - 5))
            if _valid(polished):
                polished_score = _score(polished)
                if polished_score > best_score:
                    best_result = polished
                    best_score = polished_score
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
