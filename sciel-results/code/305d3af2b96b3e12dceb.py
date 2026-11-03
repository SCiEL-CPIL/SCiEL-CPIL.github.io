import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
N_TRI = len(_indices)

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points."""
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
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

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
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

def _objective_bary(params, beta=50.0):
    """
    Objective function for optimization in barycentric coordinates.
    params is a flat array of 22 barycentric coordinates (u0,v0,u1,v1,...).
    """
    bary = params.reshape(N, 2)
    # Clip to valid range
    bary = np.clip(bary, 0.0, 1.0)
    # Ensure u+v <= 1
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask, 0] /= s[mask]
        bary[mask, 1] /= s[mask]
    
    points = _bary_to_cart(bary)
    return -_soft_min_area(points, beta)

def _objective_bary_hard(params, beta=100.0):
    """
    Harder objective for final refinement.
    """
    bary = params.reshape(N, 2)
    bary = np.clip(bary, 0.0, 1.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask, 0] /= s[mask]
        bary[mask, 1] /= s[mask]
    
    points = _bary_to_cart(bary)
    return -_min_area(points)

def _nelder_mead_bary(bary, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize in barycentric coordinates.
    """
    start_t = time.time()
    x0 = bary.flatten()
    
    result = minimize(
        _objective_bary,
        x0,
        method='Nelder-Mead',
        args=(beta,),
        options={
            'maxiter': 3000,
            'xatol': 1e-9,
            'fatol': 1e-12,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return bary, _min_area(_bary_to_cart(bary))
    
    new_bary = result.x.reshape(N, 2)
    new_bary = np.clip(new_bary, 0.0, 1.0)
    s = new_bary[:, 0] + new_bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        new_bary[mask, 0] /= s[mask]
        new_bary[mask, 1] /= s[mask]
    
    new_points = _bary_to_cart(new_bary)
    score = _min_area(new_points)
    old_score = _min_area(_bary_to_cart(bary))
    
    if score > old_score:
        return new_bary, score
    else:
        return bary, old_score

def _local_search_bary(bary, max_iter=1000, time_limit=None):
    """
    Local search in barycentric coordinates that moves points involved in smallest triangles.
    """
    if time_limit is None:
        time_limit = 30.0
    start_t = time.time()
    
    current_bary = bary.copy()
    current_points = _bary_to_cart(current_bary)
    current_min = _min_area(current_points)
    best_bary = current_bary.copy()
    best_min = current_min
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            areas = _compute_areas(current_points)
            min_area = float(np.min(areas))
            
            if min_area > best_min:
                best_min = min_area
                best_bary = current_bary.copy()
            
            # Find involved points
            threshold = min_area + 1e-10
            min_indices = np.where(areas <= threshold)[0]
            involved = set()
            for idx in min_indices:
                involved.add(int(_idx_arr[idx, 0]))
                involved.add(int(_idx_arr[idx, 1]))
                involved.add(int(_idx_arr[idx, 2]))
            
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(30):
                    du = np.random.uniform(-step_size, step_size)
                    dv = np.random.uniform(-step_size, step_size)
                    
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
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_bary, best_min

def _differential_evolution_bary(time_limit=20.0):
    """
    Use differential evolution in barycentric coordinates for global search.
    """
    start_t = time.time()
    
    # Bounds for barycentric coordinates: each u,v in [0, 1], with u+v <= 1
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def de_objective(params):
        if time.time() - start_t > time_limit:
            return 1e10  # Penalize timeout
        return _objective_bary(params, beta=30.0)
    
    result = differential_evolution(
        de_objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        popsize=15,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=True,
        init='sobol'
    )
    
    if result.success or result.fun < 1e9:
        bary = result.x.reshape(N, 2)
        bary = np.clip(bary, 0.0, 1.0)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 1.0
        if np.any(mask):
            bary[mask, 0] /= s[mask]
            bary[mask, 1] /= s[mask]
        return bary
    return None

def _sobol_bary(n, seed=42):
    """Generate n quasi-random points in barycentric coordinates."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 4
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

def _symmetric_initialization():
    """
    Generate a symmetric initial configuration.
    The equilateral triangle has 3-fold rotational symmetry.
    We place points in orbits of size 3 under rotation, plus possibly center points.
    """
    bary = np.zeros((N, 2))
    
    # 3 orbits of 3 points each = 9 points, plus 2 special points
    # Orbit 1: 3 points at radius r1 from center, angles 0, 120, 240
    # Orbit 2: 3 points at radius r2 from center, angles 60, 180, 300
    # Orbit 3: 3 points at radius r3 from center, angles 0, 120, 240 (different radius)
    # Plus 2 points: center and one more
    
    # Center in barycentric is (1/3, 1/3)
    bary[0] = [1/3, 1/3]
    
    # For a point at barycentric (u, v), the three rotations are:
    # (u, v), (v, 1-u-v), (1-u-v, u)
    
    # Orbit 1: start with (0.5, 0.2)
    u1, v1 = 0.5, 0.2
    bary[1] = [u1, v1]
    bary[2] = [v1, 1-u1-v1]
    bary[3] = [1-u1-v1, u1]
    
    # Orbit 2: start with (0.3, 0.4)
    u2, v2 = 0.3, 0.4
    bary[4] = [u2, v2]
    bary[5] = [v2, 1-u2-v2]
    bary[6] = [1-u2-v2, u2]
    
    # Orbit 3: start with (0.1, 0.1)
    u3, v3 = 0.1, 0.1
    bary[7] = [u3, v3]
    bary[8] = [v3, 1-u3-v3]
    bary[9] = [1-u3-v3, u3]
    
    # 10th point: another center-ish point
    bary[10] = [0.4, 0.3]
    
    # Clip to valid
    bary = np.clip(bary, 0.0, 1.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask, 0] /= s[mask]
        bary[mask, 1] /= s[mask]
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_bary = None
    best_score = -np.inf
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 30:
        de_bary = _differential_evolution_bary(time_limit=min(25.0, deadline - time.time() - 25))
        if de_bary is not None:
            points = _bary_to_cart(de_bary)
            score = _min_area(points)
            if score > best_score:
                best_score = score
                best_bary = de_bary
    
    # Phase 2: Multiple local optimizations from diverse starts
    initializations = []
    
    # Symmetric initialization
    sym_bary = _symmetric_initialization()
    initializations.append(sym_bary)
    
    # Sobol-based initializations
    for seed in range(8):
        sob_bary = _sobol_bary(N, seed=seed)
        initializations.append(sob_bary)
    
    # Random initializations
    for seed in range(5):
        np.random.seed(seed + 200)
        bary = np.random.dirichlet([1, 1, 1], size=N)
        initializations.append(np.column_stack([bary[:, 0], bary[:, 1]]))
    
    # If DE found a good start, include it
    if best_bary is not None:
        initializations.append(best_bary.copy())
    
    for i, init_bary in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        # Nelder-Mead optimization
        nm_bary, nm_score = _nelder_mead_bary(
            init_bary,
            time_limit=min(8.0, deadline - time.time() - 10),
            beta=50.0
        )
        
        # Local search
        ls_bary, ls_score = _local_search_bary(
            nm_bary,
            max_iter=500,
            time_limit=min(5.0, deadline - time.time() - 5)
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_bary = ls_bary
    
    # Phase 3: Final refinement with harder objective
    if best_bary is not None and time.time() < deadline - 10:
        nm_bary, nm_score = _nelder_mead_bary(
            best_bary,
            time_limit=min(8.0, deadline - time.time() - 5),
            beta=100.0
        )
        
        ls_bary, ls_score = _local_search_bary(
            nm_bary,
            max_iter=800,
            step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005],
            time_limit=min(5.0, deadline - time.time() - 2)
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_bary = ls_bary
    
    # Fallback
    if best_bary is None:
        best_bary = _sobol_bary(N, seed=0)
    
    # Final projection to ensure validity
    best_bary = np.clip(best_bary, 0.0, 1.0)
    s = best_bary[:, 0] + best_bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        best_bary[mask, 0] /= s[mask]
        best_bary[mask, 1] /= s[mask]
    
    return _bary_to_cart(best_bary)
