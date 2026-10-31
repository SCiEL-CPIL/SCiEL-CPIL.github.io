"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute all triplet indices for speed
_TRIPLETS = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
_N_TRIPLETS = len(_TRIPLETS)  # 165

def _all_areas_bary(points_bary):
    """Compute all 165 triangle areas (as ratios) for N points in barycentric coords."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    areas = np.empty(_N_TRIPLETS)
    for idx, (i, j, k) in enumerate(_TRIPLETS):
        det = (u[i] * (v[j] * w[k] - w[j] * v[k])
             - v[i] * (u[j] * w[k] - w[j] * u[k])
             + w[i] * (u[j] * v[k] - v[j] * u[k]))
        areas[idx] = abs(det)
    return areas

def _min_area_bary(points_bary):
    """Calculate the minimum area ratio of triangles formed by N points."""
    return np.min(_all_areas_bary(points_bary))

def _soft_min_area(points_bary, alpha):
    """
    Soft minimum using log-sum-exp for smoother gradients.
    """
    areas = _all_areas_bary(points_bary)
    min_val = np.min(areas)
    shifted = areas - min_val
    # For numerical stability
    log_sum = np.log(np.sum(np.exp(-shifted / alpha)))
    soft_min = min_val - alpha * log_sum
    return soft_min

def _objective(params, alpha):
    """Objective for L-BFGS-B using soft min + penalty."""
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    if np.any(w < 0):
        violations = -w[w < 0]
        penalty = 1000.0 * np.sum(violations ** 2)
    if np.any(u < 0) or np.any(v < 0):
        penalty += 1000.0 * (np.sum(u[u < 0]**2) + np.sum(v[v < 0]**2))
        
    if penalty > 1e-6:
        return penalty - 1e-4
        
    return -_soft_min_area(uv, alpha) + penalty

def _clamp_to_simplex(points):
    """Clamp points to be inside the simplex u>=0, v>=0, u+v<=1."""
    u = np.clip(points[:, 0], 0, 1)
    v = np.clip(points[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1
    if np.any(mask):
        factor = 1.0 / sum_uv[mask]
        u[mask] *= factor
        v[mask] *= factor
    return np.column_stack((u, v))

def _get_good_initializations():
    """Generate a set of good starting points."""
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Structured grid: 3x3 grid in barycentric + center
    # Barycentric grid: (i/2, j/2) where i+j <= 2, scaled
    grid_points = []
    for i in range(3):
        for j in range(3 - i):
            u = i / 2.0
            v = j / 2.0
            if u + v <= 1.0:
                grid_points.append((u, v))
    # That gives 6 points: (0,0),(0,0.5),(0,1),(0.5,0),(0.5,0.5),(1,0)
    # Add more structured points
    grid_points += [(1/3, 1/3), (2/3, 0), (0, 2/3), (1/2, 1/4), (1/4, 1/2)]
    # Pick 11 from these or generate more
    # Actually let's use a denser structured approach
    
    # Use vertices, edge midpoints, and interior points
    structured = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),  # vertices
        (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),  # edge midpoints and center-ish
        (1/3, 1/3), (2/3, 1/6), (1/6, 2/3),  # interior
        (0.25, 0.25), (0.75, 0.125)
    ]
    init1 = np.array(structured[:N])
    inits.append(init1)
    
    # 2. Perturbed versions
    for _ in range(15):
        noise = rng.uniform(-0.08, 0.08, size=(N, 2))
        perturbed = _clamp_to_simplex(init1 + noise)
        inits.append(perturbed)
        
    # 3. Random Dirichlet samples
    for _ in range(15):
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2])
    
    # 4. Symmetric patterns
    # 6 points on a hexagon-like pattern + 5 interior
    center = (1/3, 1/3)
    r = 0.25
    hex_pts = []
    for angle in range(0, 360, 60):
        rad = np.radians(angle)
        # In barycentric, moving from center
        u = center[0] + r * np.cos(rad) * 0.5
        v = center[1] + r * (np.sin(rad) * 0.5 - np.cos(rad) * 0.25)
        hex_pts.append((u, v))
    # Add 5 interior points
    interior = [(1/3, 1/3)]
    for angle in range(30, 360, 72):
        rad = np.radians(angle)
        u = center[0] + 0.1 * np.cos(rad) * 0.5
        v = center[1] + 0.1 * (np.sin(rad) * 0.5 - np.cos(rad) * 0.25)
        interior.append((u, v))
    init_hex = np.array(hex_pts[:6] + interior[:5])
    inits.append(init_hex)
    
    for _ in range(5):
        noise = rng.uniform(-0.05, 0.05, size=(N, 2))
        perturbed = _clamp_to_simplex(init_hex + noise)
        inits.append(perturbed)
        
    return inits

def _anneal_optimize(points_bary, bounds, deadline, start_alpha=1.0, end_alpha=0.01, n_steps=8):
    """
    Anneal the soft-min parameter from start_alpha to end_alpha.
    Each step runs L-BFGS-B with the current alpha.
    """
    current = points_bary.copy()
    alphas = np.geomspace(start_alpha, end_alpha, n_steps)
    
    for alpha in alphas:
        if time.time() > deadline - 2:
            break
        
        try:
            res = minimize(
                _objective,
                current.flatten(),
                args=(alpha,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < _objective(current.flatten(), alpha):
                uv_new = res.x.reshape(N, 2)
                w_new = 1 - uv_new[:, 0] - uv_new[:, 1]
                if np.all(w_new >= -1e-6) and np.all(uv_new >= -1e-6):
                    uv_new = _clamp_to_simplex(uv_new)
                    current = uv_new
        except Exception:
            pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -float('inf')
    best_points_bary = None
    
    bounds = [(0, 1)] * (2 * N)
    
    inits = _get_good_initializations()
    
    # Score and sort initializations
    init_scores = []
    for init in inits:
        score = _min_area_bary(init)
        init_scores.append((score, init))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Process top initializations with annealing
    for score, init in init_scores:
        if time.time() > deadline - 5:
            break
        
        # Annealed optimization
        result = _anneal_optimize(init, bounds, deadline, start_alpha=1.0, end_alpha=0.01, n_steps=8)
        result_score = _min_area_bary(result)
        
        if result_score > best_score:
            best_score = result_score
            best_points_bary = result.copy()
    
    if best_points_bary is None:
        best_points_bary = inits[0]
        
    # Convert barycentric to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points_bary[i]
        w = 1 - u - v
        u = max(0, u)
        v = max(0, v)
        w = max(0, w)
        s = u + v + w
        if s > 0:
            u /= s
            v /= s
            w /= s
        p = u * A + v * B + w * C
        pts.append(p)
        
    return np.array(pts)

# EVOLVE-BLOCK-END
