import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_radius(c):
    """
    Compute the feasible common radius for centres c (N,2) in [-0.5,0.5]^2.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Boundary clearance
    r_boundary = 0.5 - np.max(np.abs(c), axis=1).min()
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    
    r = min(r_boundary, r_pair)
    return max(0.0, float(r))

def _soft_min_obj(x, beta=50.0):
    """
    Objective: maximize min-pair-distance and boundary clearance.
    We minimize -r where r is approximated via a smooth min (soft-min) over
    all pair half-distances and boundary clearances.
    """
    c = x.reshape(N, 2)
    
    # Pair distances
    pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
    diff = c[pairs[:, 0]] - c[pairs[:, 1]]
    pair_dists = np.sqrt((diff ** 2).sum(axis=1)) / 2.0
    
    # Boundary clearances
    bound_clear = 0.5 - np.abs(c)
    bound_flat = bound_clear.flatten()
    
    # All values
    all_vals = np.concatenate([pair_dists, bound_flat])
    
    # Soft min: -log(sum(exp(-beta * vals))) / beta
    shifted = all_vals - all_vals.max()
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) + beta * all_vals.max()
    soft_min = -log_sum / beta
    
    return -soft_min

def _soft_min_grad(x, beta=50.0):
    """
    Gradient of the soft-min objective.
    """
    c = x.reshape(N, 2)
    
    # Pair distances
    pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
    diff = c[pairs[:, 0]] - c[pairs[:, 1]]
    pair_dists = np.sqrt((diff ** 2).sum(axis=1)) / 2.0
    
    # Boundary clearances
    bound_clear = 0.5 - np.abs(c)
    bound_flat = bound_clear.flatten()
    
    # All values
    all_vals = np.concatenate([pair_dists, bound_flat])
    
    # Gradient of soft_min w.r.t. each value
    shifted = all_vals - all_vals.max()
    exp_neg = np.exp(-beta * shifted)
    weights = exp_neg / exp_neg.sum()
    
    grad = np.zeros_like(x)
    
    # Pair part
    for k in range(len(pairs)):
        i, j = pairs[k]
        d = c[i] - c[j]
        dist = np.sqrt((d ** 2).sum())
        if dist < 1e-12:
            continue
        w = weights[k]
        grad[2 * i] += w * d / dist
        grad[2 * j] -= w * d / dist
    
    # Boundary part
    for i in range(N):
        for coord in range(2):
            idx = 2 * N + i * 2 + coord
            w = weights[idx]
            val = c[i, coord]
            if val > 0:
                grad[2 * i + coord] += w * (-1.0)
            elif val < 0:
                grad[2 * i + coord] += w * (1.0)
            else:
                grad[2 * i + coord] += 0.0
    
    return -grad

def _simulated_annealing(initial, t_start):
    """
    Perform Simulated Annealing to find a good packing.
    """
    c = initial.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.995
    max_iter = 50000
    
    for i in range(max_iter):
        if time.time() - t_start > DEADLINE - 5:
            break
        
        # Generate a random perturbation
        c_new = c + np.random.normal(0, 0.01, size=(N, 2))
        c_new = np.clip(c_new, -0.5, 0.5)
        
        new_r = _compute_radius(c_new)
        delta_r = new_r - current_r
        
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            c = c_new
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    return best_c, best_r

def _lbfgs_polish(x0, t_start):
    """
    Polish the solution using L-BFGS with soft-min objective.
    """
    if time.time() - t_start > DEADLINE - 10:
        return x0, _compute_radius(x0)
    
    try:
        result = minimize(
            _soft_min_obj,
            x0,
            method='L-BFGS-B',
            jac=_soft_min_grad,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.x is not None:
            r = _compute_radius(result.x.reshape(N, 2))
            return result.x, r
    except Exception:
        pass
    
    return x0, _compute_radius(x0)

def _hex_init():
    """
    Generate a hexagonal close-packed initial arrangement.
    """
    best_x = None
    best_r = 0
    for cols in range(6, 12):
        for rows in range(6, 12):
            if cols * rows < N:
                continue
            points = []
            for row in range(rows):
                for col in range(cols):
                    x = (col + (0.5 if row % 2 == 1 else 0)) / cols - 0.5
                    y = (row + 0.5) / rows - 0.5
                    points.append([x, y])
            if len(points) < N:
                continue
            points = np.array(points[:N])
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_x = points.copy()
    
    if best_x is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_x = grid[:N].copy()
    
    return best_x

def construct():
    t_start = time.time()
    
    # Multiple initializations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Hexagonal
    inits.append(_hex_init())
    
    # 3. Random perturbations of grid
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = inits[0] + rng.normal(0, 0.02, size=(N, 2))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        inits.append(perturbed)
    
    # 4. Random
    for _ in range(2):
        random_pts = rng.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(random_pts)
    
    best_x = None
    best_r = 0.0
    
    for init in inits:
        if time.time() - t_start > DEADLINE - 10:
            break
        
        # Simulated Annealing
        sa_result, sa_r = _simulated_annealing(init, t_start)
        
        # L-BFGS Polish
        lbfgs_result, lbfgs_r = _lbfgs_polish(sa_result.flatten(), t_start)
        
        if lbfgs_r > best_r:
            best_r = lbfgs_r
            best_x = lbfgs_result.copy()
    
    # Ensure validity
    if best_x is None:
        best_x = inits[0].flatten()
    
    result = best_x.reshape(N, 2)
    result = np.clip(result, -0.5, 0.5)
    
    return result
