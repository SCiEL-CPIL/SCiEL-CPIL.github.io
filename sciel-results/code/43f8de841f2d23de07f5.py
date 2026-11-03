"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

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

def _min_triangle_area_with_indices(points):
    """Compute min area and return the index of the minimizing triple."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    min_idx = np.argmin(areas)
    return areas[min_idx], min_idx

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

def _fast_local_optimize(bary0, max_iter=200, step_size=0.05):
    """Custom coordinate descent + Nelder-Mead hybrid for faster local optimization."""
    bary = bary0.copy()
    best_val = _min_triangle_area_vectorized(_bary_to_cart(bary))
    
    for it in range(max_iter):
        # Coordinate descent: try moving each point in each direction
        improved = False
        for i in range(N):
            for d in range(2):
                # Try positive and negative steps
                for sign in [1, -1]:
                    trial = bary.copy()
                    trial[i, d] += sign * step_size
                    trial = _clamp_bary(trial)
                    val = _min_triangle_area_vectorized(_bary_to_cart(trial))
                    if val > best_val:
                        bary = trial
                        best_val = val
                        improved = True
                    else:
                        # Try smaller step
                        trial2 = bary.copy()
                        trial2[i, d] += sign * step_size * 0.3
                        trial2 = _clamp_bary(trial2)
                        val2 = _min_triangle_area_vectorized(_bary_to_cart(trial2))
                        if val2 > best_val:
                            bary = trial2
                            best_val = val2
                            improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return bary, best_val

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
    
    # Pattern E: Based on known good configurations for Heilbronn problem
    # Use symmetry: place points at vertices, edge thirds, and interior grid
    pE = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],  # vertices
        [1/3, 0.0], [2/3, 0.0],              # bottom edge thirds
        [0.0, 1/3], [0.0, 2/3],              # left edge thirds  
        [1/3, 1/3], [2/3, 1/3],              # interior row 1
        [1/6, 1/2], [1/2, 1/2],              # interior row 2
        [5/6, 1/2]                           # interior row 2 right
    ])
    patterns.append(pE)
    
    # Pattern F: Dense interior with symmetric placement
    pF = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],  # vertices
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],  # edge midpoints
        [1/3, 1/3], [2/3, 1/3], [1/3, 2/3],  # interior triangle
        [0.5, 0.25], [0.25, 0.5], [0.75, 0.25],  # additional interior
        [0.5, 0.5]                             # center
    ])
    patterns.append(pF)
    
    # Pattern G: Random with constraint satisfaction
    pG = np.zeros((N, 2))
    # Place vertices
    pG[0] = [0.0, 0.0]
    pG[1] = [1.0, 0.0]
    pG[2] = [0.0, 1.0]
    # Place remaining 8 points randomly but well-spread
    for i in range(3, N):
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95 - u)
        pG[i] = [u, v]
    patterns.append(pG)
    
    # Choose a pattern and add small jitter
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    
    # Add small jitter
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    
    return bary

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

def _nelder_mead_optimize_bary(bary0, maxiter=200):
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

def _differential_evolution_global(seed=42, maxiter=50, popsize=15):
    """Run differential evolution for global optimization."""
    try:
        rng = np.random.RandomState(seed)
        
        # Bounds for barycentric coords
        bounds = [(0, 1)] * (2 * N)
        
        # Use a smaller population and fewer iterations for speed
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

def _hill_climb_optimize(bary0, max_iter=300, initial_step=0.03):
    """Hill climbing with adaptive step size - fast and effective for this problem."""
    bary = bary0.copy()
    best_val = _min_triangle_area_vectorized(_bary_to_cart(bary))
    step = initial_step
    
    for it in range(max_iter):
        improved = False
        
        # Try moving each point in random directions
        for _ in range(3):  # 3 attempts per iteration
            i = np.random.randint(N)
            d = np.random.randint(2)
            sign = 1 if np.random.rand() > 0.5 else -1
            
            trial = bary.copy()
            trial[i, d] += sign * step
            trial = _clamp_bary(trial)
            
            val = _min_triangle_area_vectorized(_bary_to_cart(trial))
            if val > best_val:
                bary = trial
                best_val = val
                improved = True
        
        if not improved:
            step *= 0.8
            if step < 1e-7:
                break
    
    return bary, best_val

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
    
    # Phase 1: Try multiple structured initializations with fast local optimization
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    
    for seed in initial_seeds:
        if time_left() < 5.0:
            break
        
        bary_init = _structured_init_bary(seed=seed)
        score_init = score_bary(bary_init)
        
        # Use fast hill climbing first
        if time_left() > 3.0:
            bary_opt, score_opt = _hill_climb_optimize(bary_init, max_iter=200, initial_step=0.05)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        # Then refine with L-BFGS-B if time permits
        if time_left() > 2.0:
            bary_refined = _l_bfgs_optimize_bary(bary_init, maxiter=200)
            score_refined = score_bary(bary_refined)
            if score_refined > score_init:
                score_init = score_refined
                bary_init = bary_refined
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 2: Differential evolution for global search
    if time_left() > 15.0:
        de_bary = _differential_evolution_global(seed=42, maxiter=40, popsize=15)
        if de_bary is not None:
            score_de = score_bary(de_bary)
            if score_de > best_score:
                best_score = score_de
                best_bary = de_bary.copy()
                
                # Refine with local optimization
                if time_left() > 8.0:
                    bary_refined, score_refined = _hill_climb_optimize(de_bary, max_iter=300, initial_step=0.03)
                    if score_refined > best_score:
                        best_score = score_refined
                        best_bary = bary_refined.copy()
                    if time_left() > 5.0:
                        bary_lbf = _l_bfgs_optimize_bary(best_bary, maxiter=300)
                        score_lbf = score_bary(bary_lbf)
                        if score_lbf > best_score:
                            best_score = score_lbf
                            best_bary = bary_lbf.copy()
    
    # Phase 3: Multi-start with perturbations using fast hill climbing
    rng = np.random.RandomState(123)
    restart_count = 0
    max_restarts = 100
    
    while restart_count < max_restarts and time_left() > 5.0:
        restart_count += 1
        
        # Perturb the best solution
        pert = rng.normal(0, 0.04, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        # Fast hill climbing
        bary_opt, score_opt = _hill_climb_optimize(bary_start, max_iter=150, initial_step=0.03)
        
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
    
    # Phase 4: Fine-tuning with very small perturbations
    if time_left() > 3.0:
        for ft in range(15):
            if time_left() < 1.5:
                break
            pert = rng.normal(0, 0.008, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_opt, score_opt = _hill_climb_optimize(bary_ft, max_iter=200, initial_step=0.01)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Final polish with L-BFGS-B if time remains
    if time_left() > 2.0:
        bary_final = _l_bfgs_optimize_bary(best_bary, maxiter=400)
        score_final = score_bary(bary_final)
        if score_final > best_score:
            best_score = score_final
            best_bary = bary_final.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
