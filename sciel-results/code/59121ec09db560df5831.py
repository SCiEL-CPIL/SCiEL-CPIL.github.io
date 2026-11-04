import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 100

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _soft_min_obj_and_grad(x, temp):
    """
    Smooth surrogate for maximizing the minimum of:
      - pairwise distances / 2
      - wall clearances (0.5 - |c_ik|)
    Uses log-sum-exp (soft-min) with temperature 'temp'.
    """
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-20))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    wall_clearances = 0.5 - np.abs(c)
    wall_vals = wall_clearances.flatten()
    
    pair_candidates = pair_dists * 0.5
    all_candidates = np.concatenate([pair_candidates, wall_vals])
    
    min_val = np.min(all_candidates)
    shifted = all_candidates - min_val
    exp_terms = np.exp(-shifted / temp)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_val - temp * log_sum
    
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(c)
    pair_weights = weights[:len(pair_dists)]
    ii, jj = iu
    d_safe = np.maximum(pair_dists, 1e-12)
    term_coeff = pair_weights * 0.5 / d_safe
    term_vecs = term_coeff[:, np.newaxis] * (c[ii] - c[jj])
    np.add.at(grad, ii, term_vecs)
    np.add.at(grad, jj, -term_vecs)
    
    wall_weights = weights[len(pair_dists):]
    wall_grad = -wall_weights.reshape(N, 3) * np.sign(c)
    grad += wall_grad
    
    objective = -soft_min_val
    gradient = -grad.flatten()
    
    return objective, gradient

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    """
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    np.fill_diagonal(dists, 1.0)
    
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    energy_pair = 0.5 * np.sum(violations**2)
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    wall_dists = 0.5 - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_soft_min(init_pts, temp, max_iter=5000, time_limit=30.0):
    """Run L-BFGS-B optimization on the soft-min objective with bounds."""
    t_start = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, temp)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-9, time_limit=50.0):
    """Run BFGS optimization with a time limit."""
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _generate_fcc_points(n_max=3):
    """Generate FCC lattice points."""
    points = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    # Face centers
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k])
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max + 1):
                points.append([i + 0.5, j, k + 0.5])
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    center = points.mean(axis=0)
    points = points - center
    
    # Remove duplicates
    unique_pts = []
    for p in points:
        is_dup = False
        for up in unique_pts:
            if np.linalg.norm(p - up) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_pts.append(p)
    points = np.array(unique_pts)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _lattice_objective(params, V):
    """
    Objective function for lattice optimization.
    params: [scale, tx, ty, tz, rx, ry, rz]
    V: base lattice points (N, 3)
    Returns negative score.
    """
    s, tx, ty, tz, rx, ry, rz = params
    
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    C = s * (V @ R.T) + np.array([tx, ty, tz])
    
    # Check boundaries
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    if r_bound < 0:
        return -r_bound + 10.0
    
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    r = min(r_bound, r_pair)
    return -r

def _lattice_optimize(V, time_limit=20.0):
    """Optimize lattice parameters using differential evolution."""
    t_start = time.time()
    
    bounds = [(0.05, 0.2), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
              (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    def objective(params):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _lattice_objective(params, V)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            args=(V,),
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        return result.x
    except TimeoutError:
        return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    # 1. Initialization: FCC lattice
    V_fcc = _generate_fcc_points(3)
    if len(V_fcc) >= N:
        best_pts = V_fcc.copy()
    else:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
    
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Lattice optimization with differential evolution
    if time.time() < deadline - 40.0:
        de_time = min(25.0, deadline - (time.time() - start_time) - 30.0)
        params = _lattice_optimize(V_fcc, time_limit=de_time)
        
        if params is not None:
            s, tx, ty, tz, rx, ry, rz = params
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            pts_lattice = s * (V_fcc @ R.T) + np.array([tx, ty, tz])
            pts_lattice = np.clip(pts_lattice, -0.49, 0.49)
            
            if np.all(np.isfinite(pts_lattice)):
                r_lattice = _score(pts_lattice)
                if r_lattice > best_r:
                    best_r = r_lattice
                    best_pts = pts_lattice.copy()
    
    # Strategy 2: Soft-min annealing
    temps = [0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001]
    current_pts = best_pts.copy()
    
    for temp in temps:
        if time.time() > deadline - 10.0:
            break
        
        noise = rng.normal(0, 0.003, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining = deadline - (time.time() - start_time) - 5.0
        if remaining < 5.0:
            break
        
        max_iter = 2000
        if temp < 0.01:
            max_iter = 3000
        
        opt_pts, success = _optimize_soft_min(
            init_pts, 
            temp, 
            max_iter=max_iter, 
            time_limit=remaining
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
    
    # Strategy 3: BFGS with target radius (for fine-tuning)
    if time.time() < deadline - 15.0:
        for i in range(3):
            if time.time() > deadline - 5.0:
                break
            
            target_r = best_r * 1.001  # Try to improve slightly
            noise = rng.normal(0, 0.005, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            remaining = deadline - (time.time() - start_time) - 5.0
            if remaining < 5.0:
                break
            
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                target_r, 
                max_iter=1000, 
                time_limit=remaining
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
    
    # Final Polish: Very low temperature for high precision
    if time.time() < deadline - 3.0 and best_r > 0.1:
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        remaining = deadline - (time.time() - start_time) - 2.0
        if remaining > 2.0:
            opt_pts, success = _optimize_soft_min(
                init_pts, 
                1e-8, 
                max_iter=1000, 
                time_limit=remaining
            )
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    return best_pts

# EVOLVE-BLOCK-END
