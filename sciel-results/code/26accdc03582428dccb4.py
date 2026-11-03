"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _softmin_objective(x, beta=50.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_initial_bary_grid(seed):
    """Generate a structured grid-based initial barycentric configuration."""
    rng = np.random.RandomState(seed)
    
    # Key structural points: vertices, edge midpoints, centroid
    bary_list = [
        [0.0, 0.0],       # V0
        [1.0, 0.0],       # V1
        [0.0, 1.0],       # V2
        [0.5, 0.0],       # midpoint V0-V1
        [0.5, 0.5],       # midpoint V1-V2
        [0.0, 0.5],       # midpoint V2-V0
        [1.0/3, 1.0/3],   # centroid
    ]
    
    # Add interior points in a regular pattern
    additional = [
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.75],
        [0.0, 0.25],
        [0.0, 0.75],
    ]
    
    for pt in additional:
        if len(bary_list) < N:
            if pt[0] >= 0 and pt[1] >= 0 and pt[0] + pt[1] <= 1:
                bary_list.append(pt)
    
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_list.append([u, v])
    
    return np.array(bary_list[:N])

def _make_initial_bary_random(seed):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(500):
        u = rng.uniform(0.02, 0.98)
        v = rng.uniform(0.02, 0.98)
        if u + v < 0.96:
            bary_list.append([u, v])
            if len(bary_list) == N:
                break
    while len(bary_list) < N:
        u = rng.uniform(0.05, 0.8)
        v = rng.uniform(0.05, 0.8)
        if u + v < 0.9:
            bary_list.append([u, v])
    return np.array(bary_list[:N])

def _make_initial_bary_perturb(base_bary, rng, scale):
    """Perturb a base barycentric configuration."""
    bary_pert = base_bary + rng.normal(0, scale, size=base_bary.shape)
    bary_pert = _clamp_bary(bary_pert)
    return bary_pert

def _nm_objective(x):
    """Objective for Nelder-Mead: negative min area, with projection."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _nelder_mead_optimize(x0, maxiter=200):
    """
    Custom Nelder-Mead that works on flat arrays and projects to bounds.
    """
    x = x0.copy()
    n = len(x)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x
    for i in range(n):
        dx = x.copy()
        dx[i] += 0.01
        simplex[i + 1] = _clamp_bary(dx.reshape(-1, 2)).flatten()
        
    # Evaluate initial simplex
    fvals = np.array([_nm_objective(s) for s in simplex])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(maxiter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < 1e-10:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        centroid = _clamp_bary(centroid.reshape(-1, 2)).flatten()
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = _clamp_bary(xr.reshape(-1, 2)).flatten()
        fr = _nm_objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = _clamp_bary(xe.reshape(-1, 2)).flatten()
            fe = _nm_objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = _clamp_bary(xc.reshape(-1, 2)).flatten()
            fc = _nm_objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _clamp_bary(simplex[i].reshape(-1, 2)).flatten()
                    fvals[i] = _nm_objective(simplex[i])
                    
    return simplex[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    def time_left():
        return deadline - time.time()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Start with a structured grid-based initialization
    bary_init = _make_initial_bary_grid(0)
    points_init = _bary_to_cart(bary_init)
    score_init = _min_triangle_area_vectorized(points_init)
    best_score = score_init
    best_points = points_init.copy()
    best_bary = bary_init.copy()
    
    # Generate multiple initializations: grid, random, perturbed
    inits = [bary_init]
    
    # Add random initializations
    for seed in range(1, 11):
        if time_left() < 5.0:
            break
        bary_r = _make_initial_bary_random(seed)
        inits.append(bary_r)
        
        # Also add perturbed versions of the grid
        if seed <= 5:
            rng_p = np.random.RandomState(seed + 100)
            for scale in [0.05, 0.1, 0.2]:
                if time_left() < 5.0:
                    break
                bary_p = _make_initial_bary_perturb(bary_init, rng_p, scale)
                inits.append(bary_p)
    
    # Optimize each initialization
    for i, bary0 in enumerate(inits):
        if time_left() < 10.0:
            break
            
        x0 = bary0.flatten()
        
        # First try with L-BFGS-B using the soft-min surrogate
        if time_left() > 20.0:
            try:
                result_lbfgs = minimize(
                    _softmin_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result_lbfgs.success:
                    bary_opt = _clamp_bary(result_lbfgs.x.reshape(-1, 2))
                    points_opt = _bary_to_cart(bary_opt)
                    score_opt = _min_triangle_area_vectorized(points_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_points = points_opt.copy()
                        best_bary = bary_opt.copy()
            except Exception:
                pass
        
        # Then refine with Nelder-Mead on the exact objective
        if time_left() > 15.0:
            try:
                # Use the best of the current best and the L-BFGS result
                if time_left() > 25.0:
                    # Start Nelder-Mead from the current best configuration
                    x_nm = best_bary.flatten()
                else:
                    x_nm = bary0.flatten()
                
                result_nm = _nelder_mead_optimize(x_nm, maxiter=300)
                bary_nm = _clamp_bary(result_nm.reshape(-1, 2))
                points_nm = _bary_to_cart(bary_nm)
                score_nm = _min_triangle_area_vectorized(points_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = points_nm.copy()
                    best_bary = bary_nm.copy()
            except Exception:
                pass
    
    # Final polish: run L-BFGS with higher beta on the best solution found
    if time_left() > 10.0 and best_bary is not None:
        try:
            x_final = best_bary.flatten()
            result_final = minimize(
                lambda x: _softmin_objective(x, beta=100.0),
                x_final,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result_final.success:
                bary_final = _clamp_bary(result_final.x.reshape(-1, 2))
                points_final = _bary_to_cart(bary_final)
                score_final = _min_triangle_area_vectorized(points_final)
                if score_final > best_score:
                    best_score = score_final
                    best_points = points_final.copy()
                    best_bary = bary_final.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = _bary_to_cart(bary_init)
    
    return best_points
