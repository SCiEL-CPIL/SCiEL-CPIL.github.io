import numpy as np
import scipy.optimize as opt
import scipy.spatial as spat
import time

N = 14
EPS = 1e-8

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Exact score: min triangle area / hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    # Compute all triangle areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(params, temperature=0.05):
    """
    Smooth surrogate: -softmin(areas) / hull_area
    where softmin(x) = -T * log(sum(exp(-x/T)))
    This is differentiable and approximates the minimum as T -> 0.
    
    params: array of shape (N, 2) or (2N,) representing point coordinates.
    """
    points = params.reshape(N, 2)
    n = len(points)
    
    # Compute all triangle areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft minimum: -T * log(sum(exp(-areas/T)))
    # For numerical stability, use the log-sum-exp trick
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * log_sum_exp
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10  # Large penalty
    
    # We want to maximize soft_min / hull_area, so minimize negative of that
    return -(soft_min / hull_area)

def _soft_min_grad(params, temperature=0.05):
    """
    Numerical gradient of _soft_min_score for L-BFGS-B.
    """
    n = len(params)
    grad = np.zeros(n)
    h = 1e-6
    
    for i in range(n):
        params_plus = params.copy()
        params_plus[i] += h
        params_minus = params.copy()
        params_minus[i] -= h
        grad[i] = (_soft_min_score(params_plus, temperature) - 
                   _soft_min_score(params_minus, temperature)) / (2 * h)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    # Start with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Try multiple starting points and temperatures
    start_configs = []
    
    # 1. Regular polygon
    start_configs.append(initial_points.copy())
    
    # 2. Slightly perturbed regular polygon
    for trial in range(3):
        noise = rng.normal(0, 0.1, size=(N, 2))
        perturbed = initial_points + noise
        start_configs.append(perturbed)
    
    # 3. Two-row grid arrangement (known good heuristic for Heilbronn)
    # Place points in two rows of 7
    x_coords = np.linspace(-1, 1, 7)
    row1 = np.column_stack((x_coords, np.full(7, 0.5)))
    row2 = np.column_stack((x_coords, np.full(7, -0.5)))
    grid_points = np.vstack((row1, row2))
    start_configs.append(grid_points)
    
    # 4. Hexagonal lattice subset
    hex_points = []
    for i in range(4):
        for j in range(4):
            x = i * 1.0 + (j % 2) * 0.5
            y = j * 0.866
            hex_points.append([x, y])
    hex_points = np.array(hex_points[:N])
    # Normalize to unit area
    hull = spat.ConvexHull(hex_points)
    hex_points = hex_points / np.sqrt(hull.volume)
    start_configs.append(hex_points)
    
    # Run L-BFGS-B from each starting configuration
    for start_idx, start_points in enumerate(start_configs):
        if time.time() > deadline - 20:
            break
        
        # Flatten for optimization
        x0 = start_points.flatten()
        
        # Try multiple temperature values (annealing the soft-min)
        temperatures = [0.1, 0.05, 0.02, 0.01]
        
        for temp in temperatures:
            if time.time() > deadline - 10:
                break
            
            try:
                result = opt.minimize(
                    _soft_min_score,
                    x0,
                    args=(temp,),
                    method='L-BFGS-B',
                    jac=_soft_min_grad,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
                )
                
                if result.success or result.fun < _soft_min_score(x0, temp):
                    opt_points = result.x.reshape(N, 2)
                    opt_score = _score(opt_points)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
                        # Update x0 for next temperature
                        x0 = result.x.copy()
                        
            except Exception:
                pass
        
        # After gradient-based optimization, do a few rounds of coordinate descent
        # to fine-tune
        current_points = best_points.copy()
        current_score = _score(current_points)
        
        # Simple coordinate descent: perturb one point at a time
        step_size = 0.01
        for cd_round in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                for coord in range(2):
                    for direction in [-1, 1]:
                        new_points = current_points.copy()
                        new_points[i, coord] += direction * step_size
                        
                        new_score = _score(new_points)
                        if new_score > current_score:
                            current_points = new_points
                            current_score = new_score
                            improved = True
            
            if not improved:
                break
            
            # Reduce step size
            step_size *= 0.95
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
    
    return best_points
