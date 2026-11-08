"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _grid_initial():
    """Create a grid-based initial configuration of N circles."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    while len(positions) < N:
        idx = len(positions)
        i = idx % side
        j = idx // side
        if j >= side:
            break
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.5 / side * 0.95
        positions.append([x, y, r])
    
    return np.array(positions[:N])


def _hex_initial():
    """Create a hexagonal packing initial configuration."""
    positions = []
    # Hexagonal packing: rows offset
    # For 26 circles, try to fit in a hexagonal pattern
    # Use a density-based approach
    # Approximate: sqrt(26) ~ 5.1, so about 5x5 grid with hex offset
    
    # Try different configurations
    best_hex = None
    best_score = -1
    
    for rows in range(3, 7):
        cols = int(np.ceil(N / rows))
        if cols < 3:
            continue
        
        spacing_x = 1.0 / (cols + 0.5)  # extra space for offset
        spacing_y = 1.0 / (rows + 0.5)
        
        pos = []
        for row in range(rows):
            for col in range(cols):
                if len(pos) >= N:
                    break
                x = (col + 0.5) * spacing_x + (spacing_x * 0.5 if row % 2 else 0)
                y = (row + 0.5) * spacing_y
                if x < 1.0 and y < 1.0:
                    r = min(spacing_x, spacing_y) * 0.48
                    pos.append([x, y, r])
        
        if len(pos) >= N:
            pos = pos[:N]
            # Score by sum of radii
            score = sum(p[2] for p in pos)
            if score > best_score:
                best_score = score
                best_hex = pos
    
    if best_hex is None:
        return _grid_initial()
    
    return np.array(best_hex)


def _smooth_max_0(v, eps=1e-4):
    """Smooth approximation of max(0, v) using log-sum-exp."""
    return eps * np.log(1.0 + np.exp(v / eps))

def _smooth_max_0_grad(v, eps=1e-4):
    """Gradient of smooth_max_0."""
    e = np.exp(v / eps)
    return e / (1.0 + e)

def _objective_and_gradient(params, N, penalty_weight=500.0):
    """
    Objective: minimize negative sum of radii with smooth penalty constraints.
    Uses log-sum-exp for smooth max(0, violation) to avoid kinks.
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    n = N
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    obj = -np.sum(r)
    
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    
    penalty = 0.0
    penalty_grad = np.zeros_like(params)
    
    eps = 1e-4
    
    # Boundary penalties using smooth max
    # x boundaries: r[i] - x[i] > 0 or r[i] - (1-x[i]) > 0
    viol_x_left = r - x
    viol_x_right = r - (1.0 - x)
    viol_y_bottom = r - y
    viol_y_top = r - (1.0 - y)
    
    # Smooth penalty for each boundary violation
    for viol, idx_x, idx_y, sign in [
        (viol_x_left, 0, None, 1.0),
        (viol_x_right, 1, None, -1.0),
        (viol_y_bottom, None, 0, 1.0),
        (viol_y_top, None, 1, -1.0),
    ]:
        smooth_viol = _smooth_max_0(viol, eps)
        penalty += penalty_weight * np.sum(smooth_viol)
        
        g = _smooth_max_0_grad(viol, eps)
        
        if idx_x is not None:
            # x constraint
            if idx_x == 0:  # left boundary: x < r
                penalty_grad[0::3] += penalty_weight * g * (-1.0)  # dx/d(x) = -1 for viol = r - x
                penalty_grad[2::3] += penalty_weight * g * 1.0     # dr/d(r) = 1
            else:  # right boundary: 1-x < r
                penalty_grad[0::3] += penalty_weight * g * 1.0     # dx/d(x) = 1 for viol = r - (1-x)
                penalty_grad[2::3] += penalty_weight * g * 1.0
    
        if idx_y is not None:
            # y constraint
            if idx_y == 0:  # bottom boundary: y < r
                penalty_grad[1::3] += penalty_weight * g * (-1.0)
                penalty_grad[2::3] += penalty_weight * g * 1.0
            else:  # top boundary: 1-y < r
                penalty_grad[1::3] += penalty_weight * g * 1.0
                penalty_grad[2::3] += penalty_weight * g * 1.0
    
    # Circle-circle penalties (vectorized for speed)
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx * dx + dy * dy
    min_dist = r[:, None] + r[None, :]
    
    # Only consider upper triangle (i < j)
    iu, ju = np.triu_indices(n, k=1)
    dist_sq_p = dist_sq[iu, ju]
    min_dist_p = min_dist[iu, ju]
    dx_p = dx[iu, ju]
    dy_p = dy[iu, ju]
    
    dist_p = np.sqrt(dist_sq_p + 1e-12)
    viol_p = min_dist_p - dist_p
    
    # Smooth penalty
    smooth_viol_p = _smooth_max_0(viol_p, eps)
    penalty += penalty_weight * np.sum(smooth_viol_p)
    
    g_p = _smooth_max_0_grad(viol_p, eps)
    
    # Gradient contributions
    coeff = penalty_weight * g_p / dist_p
    
    for k in range(len(iu)):
        i, j = iu[k], ju[k]
        c = coeff[k]
        penalty_grad[i] += c * dx_p[k]
        penalty_grad[j] -= c * dx_p[k]
        penalty_grad[i + n] += c * dy_p[k]
        penalty_grad[j + n] -= c * dy_p[k]
        penalty_grad[2*i + 2] += penalty_weight * g_p[k]
        penalty_grad[2*j + 2] += penalty_weight * g_p[k]
    
    total_obj = obj + penalty
    total_grad = grad + penalty_grad
    
    return total_obj, total_grad


def _check_validity(pos, radii, tol=1e-9):
    """Check if configuration is valid."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Vectorized pairwise distance check
    dx = pos[:, 0:1] - pos[:, 0:1].T
    dy = pos[:, 1:2] - pos[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, None] + radii[None, :]
    # Check upper triangle only
    iu, ju = np.triu_indices(n, k=1)
    if np.any(dist[iu, ju] < min_dist[iu, ju] - tol):
        return False
    return True


def _fix_configuration(x, y, r):
    """Fix any invalidities in the configuration."""
    n = len(r)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Clip to boundaries
    for i in range(n):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 0.001)
        x[i] = max(r[i], min(1 - r[i], x[i]))
        y[i] = max(r[i], min(1 - r[i], y[i]))
    
    # Resolve overlaps by reducing radii iteratively
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-12:
                    # Reduce the smaller radius
                    if r[i] < r[j]:
                        new_r = max(0.001, (dist - r[j]) * 0.995)
                        if new_r < r[i]:
                            r[i] = new_r
                            changed = True
                    else:
                        new_r = max(0.001, (dist - r[i]) * 0.995)
                        if new_r < r[j]:
                            r[j] = new_r
                            changed = True
        if not changed:
            break
    
    # Final boundary check with tighter tolerance
    for i in range(n):
        r[i] = min(r[i], x[i] - 1e-8, 1 - x[i] - 1e-8, y[i] - 1e-8, 1 - y[i] - 1e-8)
        r[i] = max(r[i], 0.001)
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = N
    total_params = 3 * n
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    best_obj = np.inf
    best_params = None
    best_score = 0.0
    
    def _score(params):
        """Compute the actual score (sum of radii) for a parameter set."""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        # Check validity
        if not _check_validity(np.column_stack([x, y]), r):
            return 0.0
        return np.sum(r)
    
    def _run_optimization(params_init, max_iter=500, penalty_weight=500.0):
        """Run L-BFGS-B optimization."""
        def objective_wrapper(p):
            return _objective_and_gradient(p, n, penalty_weight)[0]
        
        def jac_wrapper(p):
            return _objective_and_gradient(p, n, penalty_weight)[1]
        
        result = minimize(
            objective_wrapper,
            params_init,
            jac=jac_wrapper,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30,
                     'maxfun': 3000}
        )
        
        return result
    
    def _high_precision_polish(params_init, max_iter=1000):
        """Final polish with very tight tolerances."""
        def objective_wrapper(p):
            return _objective_and_gradient(p, n, penalty_weight=5000.0)[0]
        
        def jac_wrapper(p):
            return _objective_and_gradient(p, n, penalty_weight=5000.0)[1]
        
        result = minimize(
            objective_wrapper,
            params_init,
            jac=jac_wrapper,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20,
                     'maxfun': 5000}
        )
        
        return result
    
    # Try multiple initial configurations
    configs = []
    
    # 1. Grid initial
    init_grid = _grid_initial()
    configs.append(init_grid.flatten())
    
    # 2. Hex initial
    init_hex = _hex_initial()
    configs.append(init_hex.flatten())
    
    # 3. Random restarts with better distribution
    np.random.seed(42)
    for restart in range(12):
        # Use different strategies for random init
        if restart % 3 == 0:
            # Uniform random
            x_rand = np.random.uniform(0.1, 0.9, size=n)
            y_rand = np.random.uniform(0.1, 0.9, size=n)
            r_rand = np.random.uniform(0.02, 0.09, size=n)
        elif restart % 3 == 1:
            # Clustered around grid points
            side = int(np.ceil(np.sqrt(n)))
            x_rand = (np.random.randint(0, side, size=n) + 0.5) / side + np.random.normal(0, 0.05, size=n)
            y_rand = (np.random.randint(0, side, size=n) + 0.5) / side + np.random.normal(0, 0.05, size=n)
            r_rand = np.random.uniform(0.03, 0.08, size=n)
        else:
            # Corner-heavy
            corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
            x_rand = np.random.uniform(0.1, 0.9, size=n)
            y_rand = np.random.uniform(0.1, 0.9, size=n)
            for i in range(min(4, n)):
                cx, cy = corners[i % 4]
                x_rand[i] = cx + np.random.normal(0, 0.1)
                y_rand[i] = cy + np.random.normal(0, 0.1)
            r_rand = np.random.uniform(0.03, 0.08, size=n)
        
        x_rand = np.clip(x_rand, 0.05, 0.95)
        y_rand = np.clip(y_rand, 0.05, 0.95)
        
        params_rand = np.zeros(3*n)
        params_rand[0::3] = x_rand
        params_rand[1::3] = y_rand
        params_rand[2::3] = r_rand
        configs.append(params_rand)
    
    # Run optimization on each config
    for idx, params_init in enumerate(configs):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        try:
            result = _run_optimization(params_init, max_iter=400)
        except Exception:
            continue
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
        
        # Compute actual score
        score = _score(result.x)
        if score > best_score:
            best_score = score
            best_params = result.x.copy()
        
        # If we have time, do a polish step on the best so far
        if time.time() < deadline - 15 and best_params is not None:
            try:
                result2 = _run_optimization(best_params, max_iter=300)
                score2 = _score(result2.x)
                if score2 > best_score:
                    best_score = score2
                    best_params = result2.x.copy()
            except Exception:
                pass
    
    # High precision polish if we have time
    if time.time() < deadline - 10 and best_params is not None:
        try:
            result_polish = _high_precision_polish(best_params, max_iter=500)
            score_polish = _score(result_polish.x)
            if score_polish > best_score:
                best_score = score_polish
                best_params = result_polish.x.copy()
        except Exception:
            pass
    
    # Another round of polish if still time
    if time.time() < deadline - 5 and best_params is not None:
        try:
            result_polish2 = _high_precision_polish(best_params, max_iter=300)
            score_polish2 = _score(result_polish2.x)
            if score_polish2 > best_score:
                best_score = score_polish2
                best_params = result_polish2.x.copy()
        except Exception:
            pass
    
    # Final fix and validation
    if best_params is None:
        # Fallback to grid
        x, y, r = _grid_initial().T
    else:
        x = best_params[0::3]
        y = best_params[1::3]
        r = best_params[2::3]
    
    x, y, r = _fix_configuration(x, y, r)
    
    result_array = np.column_stack([x, y, r])
    
    return result_array

# EVOLVE-BLOCK-END
