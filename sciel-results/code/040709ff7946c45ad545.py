import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    # Wall clearance: min over i,k of (0.5 - |c_ik|)
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    # Inter-sphere clearance: min over i<j of |c_i - c_j| / 2
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def _smooth_min(values, beta=100.0):
    """Compute a smooth approximation of the minimum using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    shifted = -beta * (values - max_val)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val - log_sum / beta

def _objective(x, beta=100.0):
    """Smooth objective: negative of soft-min clearance."""
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Wall clearance
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    wall_vals = 0.5 - max_coord
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 0.0
    
    return -_smooth_min(all_vals, beta)

def _objective_grad(x, beta=100.0):
    """Gradient of the smooth objective."""
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Wall clearance
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    wall_vals = 0.5 - max_coord
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return np.zeros_like(x)
    
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    shifted = np.clip(shifted, -700, 0)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient with respect to pair values
    grad_pair = np.zeros_like(pair_vals)
    grad_pair_flat = -weights[:len(pair_vals.ravel())] / beta
    grad_pair = grad_pair_flat.reshape(pair_vals.shape)
    
    # Gradient with respect to wall values
    grad_wall = -weights[len(pair_vals.ravel()):] / beta
    
    # Now compute actual gradient
    grad = np.zeros_like(x)
    
    # Pairwise contribution
    for i in range(N):
        for j in range(i+1, N):
            w = grad_pair[i, j]
            if abs(w) > 1e-12:
                d = c[i] - c[j]
                dist = np.sqrt(np.sum(d**2))
                if dist > 1e-12:
                    grad[i] += w * d / dist
                    grad[j] -= w * d / dist
    
    # Wall contribution
    for i in range(N):
        for k in range(3):
            if np.abs(c[i, k]) > 1e-12:
                sign = np.sign(c[i, k])
                grad[i, k] += grad_wall[i] * sign
    
    return grad

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def make_initial_guess(seed, lattice_type='cubic'):
        rng = np.random.RandomState(seed)
        
        if lattice_type == 'cubic':
            # 5x5x5 grid
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            grid = grid[:N]
            jitter = rng.uniform(-0.01, 0.01, size=(N, 3))
            grid = grid + jitter
            grid = np.clip(grid, -0.49, 0.49)
            return grid.flatten()
        
        elif lattice_type == 'fcc':
            # FCC lattice points
            points = []
            s = 0.4  # spacing
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        # FCC: corners + face centers
                        points.append([i*s, j*s, k*s])
                        if i < 2:
                            points.append([(i+0.5)*s, j*s, k*s])
                        if j < 2:
                            points.append([i*s, (j+0.5)*s, k*s])
                        if k < 2:
                            points.append([i*s, j*s, (k+0.5)*s])
            points = np.array(points)
            # Filter to cube
            mask = np.all(np.abs(points) < 0.49, axis=1)
            points = points[mask]
            if len(points) < N:
                # Add random points
                extra = rng.uniform(-0.49, 0.49, size=(N - len(points), 3))
                points = np.vstack([points, extra])
            else:
                points = points[:N]
            jitter = rng.uniform(-0.005, 0.005, size=(N, 3))
            points = points + jitter
            points = np.clip(points, -0.49, 0.49)
            return points.flatten()
        
        elif lattice_type == 'random':
            points = rng.uniform(-0.49, 0.49, size=(N, 3))
            return points.flatten()
        
        else:
            # Default: 5x5x5 grid
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            grid = grid[:N]
            jitter = rng.uniform(-0.01, 0.01, size=(N, 3))
            grid = grid + jitter
            grid = np.clip(grid, -0.49, 0.49)
            return grid.flatten()
    
    best_r = 0.0
    best_c = None
    
    # Try multiple initial configurations and lattice types
    configs = [
        (0, 'cubic'),
        (1, 'cubic'),
        (2, 'fcc'),
        (3, 'fcc'),
        (4, 'random'),
        (5, 'cubic'),
    ]
    
    for seed, lattice_type in configs:
        if time.time() - start_time > time_limit - 20:
            break
        
        x0 = make_initial_guess(seed, lattice_type)
        x_current = x0.copy()
        
        # Phase 1: Coarse optimization with low beta
        res1 = minimize(
            lambda x: _objective(x, beta=20.0),
            x_current,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfev': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res1.success or res1.fun < 1e-4:
            x_current = res1.x
        
        # Phase 2: Medium optimization
        if time.time() - start_time > time_limit - 15:
            c_test = x_current.reshape(N, 3)
            r_test = _compute_radius(c_test)
            if r_test > best_r:
                best_r = r_test
                best_c = c_test.copy()
            continue
        
        res2 = minimize(
            lambda x: _objective(x, beta=50.0),
            x_current,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfev': 1500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res2.success or res2.fun < 1e-6:
            x_current = res2.x
        
        # Phase 3: Fine optimization with high beta
        if time.time() - start_time > time_limit - 10:
            c_test = x_current.reshape(N, 3)
            r_test = _compute_radius(c_test)
            if r_test > best_r:
                best_r = r_test
                best_c = c_test.copy()
            continue
        
        res3 = minimize(
            lambda x: _objective(x, beta=100.0),
            x_current,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfev': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res3.success or res3.fun < 1e-8:
            x_current = res3.x
        
        # Phase 4: Final polish with very high beta
        if time.time() - start_time > time_limit - 5:
            c_test = x_current.reshape(N, 3)
            r_test = _compute_radius(c_test)
            if r_test > best_r:
                best_r = r_test
                best_c = c_test.copy()
            continue
        
        res4 = minimize(
            lambda x: _objective(x, beta=200.0),
            x_current,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfev': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if res4.success or res4.fun < 1e-10:
            x_current = res4.x
        
        # Evaluate final result
        c_test = x_current.reshape(N, 3)
        r_test = _compute_radius(c_test)
        
        if r_test > best_r:
            best_r = r_test
            best_c = c_test.copy()
    
    # If we have time, try a few more random restarts
    if best_c is None or time.time() - start_time < time_limit - 30:
        for extra_seed in range(6, 12):
            if time.time() - start_time > time_limit - 15:
                break
            
            x0 = make_initial_guess(extra_seed, 'random')
            
            res = minimize(
                lambda x: _objective(x, beta=100.0),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfev': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < 1e-8:
                c_test = res.x.reshape(N, 3)
                r_test = _compute_radius(c_test)
                
                if r_test > best_r:
                    best_r = r_test
                    best_c = c_test.copy()
    
    if best_c is None:
        # Fallback to simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
