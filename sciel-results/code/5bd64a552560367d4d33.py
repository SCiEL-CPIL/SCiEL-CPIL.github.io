import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - log_sum / beta

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points.
    bary has shape (N, 2) with u, v >= 0 and u + v <= 1.
    u = weight of A=(0,0), v = weight of B=(1,0), 1-u-v = weight of C=(0.5,H)
    """
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _project_to_simplex(bary):
    """Project barycentric coordinates onto the simplex u,v >= 0, u+v <= 1."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _is_valid(points):
    """Check if all points are inside the triangle."""
    eps = 1e-8
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    if np.any(y > SQRT3 * x + eps):
        return False
    if np.any(y > -SQRT3 * x + SQRT3 + eps):
        return False
    return True

def _bary_objective(bary_flat, beta=50.0):
    """
    Objective function for optimization: negative of soft min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = _project_to_simplex(bary)
    points = _bary_to_cart(bary)
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize(bary, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective in barycentric coords.
    """
    start_t = time.time()
    x0 = bary.flatten()
    
    try:
        result = minimize(
            _bary_objective,
            x0,
            method='Nelder-Mead',
            args=(beta,),
            options={
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
    except Exception:
        return bary, _min_area(_bary_to_cart(bary))
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return bary, _min_area(_bary_to_cart(bary))
    
    new_bary = _project_to_simplex(result.x.reshape(N, 2))
    new_points = _bary_to_cart(new_bary)
    
    if not _is_valid(new_points):
        return bary, _min_area(_bary_to_cart(bary))
    
    score = _min_area(new_points)
    old_score = _min_area(_bary_to_cart(bary))
    
    if score > old_score:
        return new_bary, score
    else:
        return bary, old_score

def _local_search(bary, time_limit=5.0, rng=None):
    """
    Local search that moves points involved in the smallest triangles.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_bary = _project_to_simplex(bary.copy())
    current_points = _bary_to_cart(current_bary)
    current_min = _min_area(current_points)
    best_bary = current_bary.copy()
    best_min = current_min
    
    start_t = time.time()
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(200):
            if time.time() - start_t > time_limit:
                break
            
            areas = _compute_areas(current_points)
            min_area = float(np.min(areas))
            
            if min_area > best_min:
                best_min = min_area
                best_bary = current_bary.copy()
                
            # Find points involved in min triangles
            threshold = min_area + 1e-10
            min_indices = np.where(areas <= threshold)[0]
            involved = set()
            for idx in min_indices:
                involved.add(int(_idx_arr[idx, 0]))
                involved.add(int(_idx_arr[idx, 1]))
                involved.add(int(_idx_arr[idx, 2]))
            
            if len(involved) == 0:
                break
            
            improved = False
            for idx in list(involved):
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(20):
                    du = rng.uniform(-step_size, step_size)
                    dv = rng.uniform(-step_size, step_size)
                    
                    new_u = current_bary[idx, 0] + du
                    new_v = current_bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                        
                    test_bary = current_bary.copy()
                    test_bary[idx, 0] = new_u
                    test_bary[idx, 1] = new_v
                    test_points = _bary_to_cart(test_bary)
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_bary = test_bary
                        current_points = test_points
                        min_area = test_min
                        improved = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                break
                
    return best_bary, best_min

def _initial_guess():
    """Generate a structured initial guess based on a grid pattern in barycentric coords."""
    positions = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.33, 0.33), (0.67, 0.33), (0.33, 0.67),
        (0.5, 0.25), (0.25, 0.5), (0.75, 0.25),
        (0.4, 0.4), (0.6, 0.4),
    ]
    bary = np.empty((N, 2))
    for m, (u, v) in enumerate(positions):
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary[m, 0] = u
        bary[m, 1] = v
    return bary

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def _sobol_barycentric(n, seed=42):
    """Generate n quasi-random points in the simplex using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 6
    samples = rng.random(n_gen)
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    if len(valid_u) >= n:
        u = valid_u[:n]
        v = valid_v[:n]
    else:
        while len(valid_u) < n:
            more = rng.random(n_gen)
            u2, v2 = more[:, 0], more[:, 1]
            mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
            valid_u = np.concatenate([valid_u, u2[mask2]])
            valid_v = np.concatenate([valid_v, v2[mask2]])
        u = valid_u[:n]
        v = valid_v[:n]
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -np.inf
    
    def _try_optimize(bary, beta, time_limit):
        """Try to optimize from initial barycentric points, return (bary, score) or (None, -inf)."""
        try:
            bary_out, score = _nelder_mead_optimize(bary, time_limit=time_limit, beta=beta)
            bary_out = _project_to_simplex(bary_out)
            pts = _bary_to_cart(bary_out)
            if _is_valid(pts):
                score = _min_area(pts)
                return bary_out, score
        except Exception:
            pass
        return None, -np.inf
    
    # Phase 1: Annealing schedule with Nelder-Mead from structured guess
    initial_bary = _initial_guess()
    beta_schedule = [10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
    
    current_bary = initial_bary.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        time_limit = min(12.0, remaining * 0.3)
        
        if time_limit < 5.0:
            break
        
        bary_out, score = _try_optimize(current_bary, beta=beta, time_limit=time_limit)
        if bary_out is not None and score > best_score:
            best_bary = bary_out
            best_score = score
            current_bary = bary_out
    
    # Apply local search to the best result
    if best_bary is not None and time.time() < deadline - 25:
        nudged_bary, nudged_score = _local_search(best_bary, time_limit=10.0, rng=rng)
        if nudged_score > best_score:
            best_bary = nudged_bary
            best_score = nudged_score

    # Phase 2: Multiple restarts from diverse initial points
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        if i % 2 == 0:
            random_bary = _random_valid_barycentric(rng)
        else:
            random_bary = _sobol_barycentric(N, seed=42 + i * 13)
        
        beta = 30.0 + 10.0 * (i % 6)
        
        bary_out, score = _try_optimize(random_bary, beta=beta, time_limit=7.0)
        if bary_out is not None and score > best_score:
            best_bary = bary_out
            best_score = score
            
        if bary_out is not None and time.time() < deadline - 15:
            nudged_bary, nudged_score = _local_search(bary_out, time_limit=4.0, rng=rng)
            if nudged_score > best_score:
                best_bary = nudged_bary
                best_score = nudged_score
    
    # Phase 3: High-precision refinement
    if best_bary is not None:
        if time.time() < deadline - 12:
            bary_out, score = _try_optimize(best_bary, beta=500.0, time_limit=8.0)
            if bary_out is not None and score > best_score:
                best_bary = bary_out
                best_score = score
        
        if time.time() < deadline - 8:
            nudged_bary, nudged_score = _local_search(best_bary, time_limit=6.0, rng=rng)
            if nudged_score > best_score:
                best_bary = nudged_bary
                best_score = nudged_score
        
        if time.time() < deadline - 5:
            bary_out, score = _try_optimize(best_bary, beta=1000.0, time_limit=4.0)
            if bary_out is not None and score > best_score:
                best_bary = bary_out
                best_score = score
    
    # Phase 4: If we still have time, try more diverse restarts with different betas
    if best_bary is not None and time.time() < deadline - 15:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            random_bary = _random_valid_barycentric(rng)
            beta = 15.0 + 25.0 * (i % 4)
            
            bary_out, score = _try_optimize(random_bary, beta=beta, time_limit=6.0)
            if bary_out is not None and score > best_score:
                best_bary = bary_out
                best_score = score
                
            if bary_out is not None and time.time() < deadline - 8:
                nudged_bary, nudged_score = _local_search(bary_out, time_limit=3.0, rng=rng)
                if nudged_score > best_score:
                    best_bary = nudged_bary
                    best_score = nudged_score
    
    # Final check and strict validation
    if best_bary is None:
        bary = _random_valid_barycentric(rng)
        best_bary = bary
    
    final_bary = _project_to_simplex(best_bary)
    final_points = _bary_to_cart(final_bary)
    
    return final_points
