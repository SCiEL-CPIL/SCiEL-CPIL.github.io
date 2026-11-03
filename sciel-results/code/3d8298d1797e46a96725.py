"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

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
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
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

def _clamp_cart(points):
    """Clamp Cartesian points to be inside the triangle."""
    bary = _cart_to_bary(points)
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)

def _objective_bary(x):
    """Objective in barycentric coordinates: negative min triangle area.
    Barycentric coords (u,v) with u>=0, v>=0, u+v<=1 naturally constrain to triangle."""
    bary = x.reshape(N, 2)
    # Clamp to valid region
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _structured_init_bary(seed=42):
    """Generate diverse structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    # Pattern 1: Vertices + edge midpoints + centroid + interior
    patterns = []
    
    # Pattern A: Symmetric grid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread out
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Random-ish structured
    pD = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [rng.uniform(0.3, 0.7), 0.0],
        [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)],
        [0.0, rng.uniform(0.3, 0.7)],
        [1/3, 1/3],
        [rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.1, 0.3), rng.uniform(0.7, 0.9)],
        [rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)]
    ])
    patterns.append(pD)
    
    # Pattern E: Concentric rings
    pE = np.array([
        [1/3, 1/3],
        [1/6, 1/6], [5/6, 1/6], [1/6, 5/6],
        [0.2, 0.0], [0.8, 0.0], [0.0, 0.2],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75],
        [0.5, 0.5]
    ])
    patterns.append(pE)
    
    # Pattern F: Edge-biased
    pF = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.3, 0.0], [0.7, 0.0],
        [0.0, 0.3], [0.0, 0.7],
        [0.2, 0.2], [0.5, 0.3], [0.3, 0.5],
        [0.5, 0.5]
    ])
    patterns.append(pF)
    
    # Choose a pattern and add small jitter
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    
    # Add small jitter
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    
    return bary

def _random_init_bary(seed):
    """Random points in barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary = []
    for _ in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1 - u)
        bary.append([u, v])
    return np.array(bary)

def _grid_init_bary(level, seed=42):
    """Deterministic grid with structured jitter in barycentric space."""
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(level + 1):
        for j in range(level + 1 - i):
            u = i / level
            v = j / level
            if u + v <= 1.0:
                # Jitter in barycentric space
                ju = rng.uniform(-0.03, 0.03)
                jv = rng.uniform(-0.03, 0.03)
                bu, bv = u + ju, v + jv
                bu = max(0.0, min(bu, 1.0))
                bv = max(0.0, min(bv, 1.0))
                if bu + bv > 1.0:
                    scale = 1.0 / (bu + bv)
                    bu *= scale
                    bv *= scale
                pts.append([bu, bv])
    
    if len(pts) > N:
        idx = rng.choice(len(pts), size=N, replace=False)
        pts = [pts[i] for i in idx]
    elif len(pts) < N:
        while len(pts) < N:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1 - u)
            pts.append([u, v])
    return np.array(pts)

def _nelder_mead_optimize_bary(bary0, maxiter=2000):
    """Run Nelder-Mead optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _objective_bary, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _l_bfgs_optimize_bary(bary0, maxiter=500):
    """Run L-BFGS-B optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        
        # Bounds: u in [0,1], v in [0,1] (u+v<=1 handled by clamping)
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _differential_evolution_global(seed=42, maxiter=50, popsize=15):
    """Run differential evolution for global optimization."""
    try:
        # Bounds for barycentric coords
        bounds = [(0, 1)] * (2 * N)
        
        de_result = differential_evolution(
            _objective_bary,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            updating='deferred'
        )
        
        bary_opt = _clamp_bary(de_result.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    def score_bary(bary):
        points = _bary_to_cart(bary)
        return _min_triangle_area_vectorized(points)
    
    def update_best(bary):
        nonlocal best_bary, best_score
        bary = _clamp_bary(bary)
        s = score_bary(bary)
        if s > best_score:
            best_score = s
            best_bary = bary.copy()
    
    # Phase 1: Differential Evolution for global search
    if time_left() > 30.0:
        de_bary = _differential_evolution_global(seed=42, maxiter=60, popsize=12)
        if de_bary is not None:
            update_best(de_bary)
    
    # Phase 2: Multi-start local optimization from diverse initializations
    inits = []
    
    # Structured patterns
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]:
        inits.append(_structured_init_bary(seed=seed))
    
    # Grid-based initializations
    for level in [2, 3, 4, 5, 6]:
        for seed in [42, 123, 7, 999, 2024]:
            inits.append(_grid_init_bary(level, seed))
    
    # Random initializations
    for seed in [42, 123, 7, 999, 2024, 31415, 27182, 16180, 14142, 17320]:
        inits.append(_random_init_bary(seed))
    
    # If DE found a good solution, add perturbed versions
    if best_bary is not None:
        rng = np.random.RandomState(777)
        for scale in [0.03, 0.05, 0.1, 0.15]:
            pert = best_bary + rng.uniform(-scale, scale, size=(N, 2))
            pert = _clamp_bary(pert)
            inits.append(pert)
    
    # Run local optimization from each initialization
    for idx, init_bary in enumerate(inits):
        if time_left() < 5.0:
            break
        
        # Try Nelder-Mead (robust for non-smooth objective)
        bary_opt = _nelder_mead_optimize_bary(init_bary, maxiter=1500)
        update_best(bary_opt)
        
        # If time remains, try L-BFGS-B from the NM result (faster convergence near optimum)
        if time_left() > 8.0:
            bary_lbfgs = _l_bfgs_optimize_bary(bary_opt, maxiter=300)
            update_best(bary_lbfgs)
    
    # Phase 3: Final refinement on best solution if time remains
    if best_bary is not None and time_left() > 3.0:
        bary_final = _nelder_mead_optimize_bary(best_bary, maxiter=3000)
        update_best(bary_final)
        
        if time_left() > 2.0:
            bary_final2 = _l_bfgs_optimize_bary(bary_final, maxiter=500)
            update_best(bary_final2)
    
    if best_bary is None:
        # Fallback: simple random
        best_bary = _random_init_bary(0)
    
    return _bary_to_cart(best_bary)
