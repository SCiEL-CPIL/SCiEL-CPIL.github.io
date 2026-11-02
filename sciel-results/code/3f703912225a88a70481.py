import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    # Find all triangles with area close to min
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    # Collect unique point indices involved
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _soft_min_area(points, temperature=0.005):
    """
    Smooth approximation of min area using log-sum-exp.
    soft_min(a) = -T * log(sum(exp(-a_i / T)))
    """
    areas = _compute_areas(points)
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def _objective(params, temperature=0.005):
    """
    Objective to minimize: negative soft-min area.
    params: 22 values representing barycentric coordinates (u, v) for each point.
    """
    uv = params.reshape(N, 2)
    u = np.clip(uv[:, 0], 0.0, None)
    v = np.clip(uv[:, 1], 0.0, None)
    
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    
    soft_min = _soft_min_area(pts, temperature)
    return -soft_min

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
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

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        # u + v = 1 - y/H
        # v - u = 2*x - 1
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
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

def _initial_guess_grid():
    """Generate a structured initial guess based on a grid pattern."""
    positions = [
        (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),
        (0.2, 0.3), (0.5, 0.3), (0.8, 0.3),
        (0.15, 0.5), (0.45, 0.5), (0.75, 0.5),
        (0.3, 0.7), (0.6, 0.7),
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

def _local_nudge_search(points, max_iter=500, time_limit=10.0, rng=None):
    """
    Local search that only moves points involved in the smallest triangles.
    Uses targeted perturbations to break near-degeneracies.
    Returns improved points and score.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
            
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
                
            if len(involved) == 0:
                break
                
            # Try nudging each involved point in random directions
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                    
                # Try several random perturbations for this point
                for _ in range(20):
                    du = rng.uniform(-step_size, step_size)
                    dv = rng.uniform(-step_size, step_size)
                    
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                        
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    # Check if inside triangle
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                        
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        min_area = test_min
                        improved = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                break
                
    return best_points, best_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Bounds for barycentric coordinates: each in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    def _try_optimize(initial_params, temperature, maxiter, method='L-BFGS-B'):
        """Try to optimize from initial params, return (points, score) or (None, -inf)."""
        try:
            result = minimize(
                _objective,
                initial_params,
                args=(temperature,),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < -np.inf + 1:
                pts = _bary_to_cart(result.x.reshape(N, 2))
                # Verify validity
                if np.all(pts[:, 0] >= -1e-8) and np.all(pts[:, 0] <= 1.0 + 1e-8) and \
                   np.all(pts[:, 1] >= -1e-8) and np.all(pts[:, 1] <= H + 1e-8):
                    x = pts[:, 0]
                    y = pts[:, 1]
                    if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and \
                       np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
                        score = _min_area(pts)
                        return pts, score
        except Exception:
            pass
        return None, -np.inf
    
    # Phase 1: Multiple restarts with L-BFGS-B using smooth objective
    
    # Start with a good structured guess
    initial_bary = _initial_guess_grid()
    initial_params = initial_bary.flatten()
    
    pts, score = _try_optimize(initial_params, temperature=0.01, maxiter=300)
    if pts is not None and score > best_score:
        best_points = pts
        best_score = score
        
    # Apply local nudge search to the initial result
    if best_points is not None and time.time() < deadline - 40:
        nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=300, time_limit=15.0, rng=rng)
        if nudged_score > best_score:
            best_points = nudged_pts
            best_score = nudged_score

    # Additional restarts from random initial points
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate random initial barycentric coords
        random_bary = _random_valid_barycentric(rng)
        random_params = random_bary.flatten()
        
        # Use slightly different temperatures
        temp = 0.01 + 0.005 * i
        
        pts, score = _try_optimize(random_params, temperature=temp, maxiter=200)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
        # Apply local nudge search to each random restart result
        if pts is not None and time.time() < deadline - 20:
            nudged_pts, nudged_score = _local_nudge_search(pts, max_iter=200, time_limit=5.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score
    
    # Phase 2: Refine best result with lower temperature for higher precision
    if best_points is not None:
        if time.time() > deadline - 15:
            return best_points
        
        # Convert best points back to barycentric for refinement
        best_bary = _cart_to_bary(best_points)
        best_bary = np.clip(best_bary, 0.0, 1.0)
        for m in range(N):
            if best_bary[m, 0] + best_bary[m, 1] > 1.0:
                s = 1.0 / (best_bary[m, 0] + best_bary[m, 1])
                best_bary[m, 0] *= s
                best_bary[m, 1] *= s
        
        best_params = best_bary.flatten()
        
        # Refine with lower temperature
        pts, score = _try_optimize(best_params, temperature=0.003, maxiter=500)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
        # Apply nudge search to refined result
        if time.time() < deadline - 10:
            nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=400, time_limit=10.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score

        # Further refinement with even lower temperature
        if time.time() > deadline - 10:
            return best_points
        
        best_bary2 = _cart_to_bary(best_points)
        best_bary2 = np.clip(best_bary2, 0.0, 1.0)
        for m in range(N):
            if best_bary2[m, 0] + best_bary2[m, 1] > 1.0:
                s = 1.0 / (best_bary2[m, 0] + best_bary2[m, 1])
                best_bary2[m, 0] *= s
                best_bary2[m, 1] *= s
        
        pts, score = _try_optimize(best_bary2.flatten(), temperature=0.001, maxiter=1000)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
        # Final nudge search
        if time.time() < deadline - 5:
            nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=500, time_limit=5.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score
    
    # Phase 3: If we still have time, try differential evolution for global search
    if best_points is None or time.time() < deadline - 30:
        if time.time() > deadline - 25:
            if best_points is not None:
                return best_points
            # Fallback: return a valid configuration
            bary = _random_valid_barycentric(rng)
            return _bary_to_cart(bary)
        
        def de_objective(params):
            return _objective(params, temperature=0.01)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='sobol'
            )
            
            if de_result.fun < -np.inf + 1:
                de_bary = de_result.x.reshape(N, 2)
                de_bary = np.clip(de_bary, 0.0, 1.0)
                for m in range(N):
                    if de_bary[m, 0] + de_bary[m, 1] > 1.0:
                        s = 1.0 / (de_bary[m, 0] + de_bary[m, 1])
                        de_bary[m, 0] *= s
                        de_bary[m, 1] *= s
                
                de_pts = _bary_to_cart(de_bary)
                de_score = _min_area(de_pts)
                
                if de_score > best_score:
                    best_points = de_pts
                    best_score = de_score
                    
                # Refine DE result
                if time.time() < deadline - 10:
                    pts, score = _try_optimize(de_bary.flatten(), temperature=0.005, maxiter=300)
                    if pts is not None and score > best_score:
                        best_points = pts
                        best_score = score
                        
                    # Nudge search on DE result
                    if time.time() < deadline - 5:
                        nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=300, time_limit=5.0, rng=rng)
                        if nudged_score > best_score:
                            best_points = nudged_pts
                            best_score = nudged_score
        except Exception:
            pass
    
    # Final check and strict validation
    if best_points is None:
        # Fallback: return a valid configuration
        bary = _random_valid_barycentric(rng)
        best_points = _bary_to_cart(bary)
    
    # Strictly enforce containment by re-clipping the final output
    final_bary = _cart_to_bary(best_points)
    final_points = _bary_to_cart(final_bary)
    
    return final_points
