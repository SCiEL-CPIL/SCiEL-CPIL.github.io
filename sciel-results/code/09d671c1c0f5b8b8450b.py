"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Combined strategy: Soft-Min Surrogate + Beta Annealing + Differential Evolution.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples (C(11,3) = 165)
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _softmin_objective(x, beta=100.0):
    """
    Smooth surrogate: negative soft-min of triangle areas using log-sum-exp.
    This makes the objective differentiable, allowing gradient-based optimizers to work.
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min calculation: min(a) approx -1/beta * log(sum(exp(-beta*a)))
    # To avoid overflow, shift by min area
    min_area = np.min(areas)
    shifted = areas - min_area
    # Use log-sum-exp trick for stability
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_area - log_sum / beta
    return -soft_min

def _hard_objective(x):
    """Hard objective: negative min triangle area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _score_bary(bary):
    points = _bary_to_cart(bary)
    return _min_triangle_area_vectorized(points)

def _generate_initial_bary(seed, mode=0):
    """Generate diverse initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    if mode == 0:
        # Symmetric pattern with vertices, edge points, centroid
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
        ])
    elif mode == 1:
        # Hexagonal-ish
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [0.33, 0.33],
            [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
        ])
    elif mode == 2:
        # Spread pattern
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
            [0.33, 0.33],
            [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
        ])
    elif mode == 3:
        # Random structured
        bary = np.array([
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
    elif mode == 4:
        # Pure random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.01, 0.99)
                v = rng.uniform(0.01, 0.99)
                if u + v < 0.98:
                    bary[i] = [u, v]
                    break
    else:
        # Default to random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.01, 0.99)
                v = rng.uniform(0.01, 0.99)
                if u + v < 0.98:
                    bary[i] = [u, v]
                    break
    
    # Add small jitter for diversity
    jitter = rng.normal(0, 0.015, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _l_bfgs_softmin(bary0, maxiter=300, beta=100.0):
    """L-BFGS-B optimization using soft-min surrogate."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _softmin_objective(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _l_bfgs_hard(bary0, maxiter=150):
    """L-BFGS-B optimization using hard min."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _hard_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _nelder_mead_optimize(bary0, maxiter=200):
    """Nelder-Mead for local refinement on hard objective."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _beta_anneal_refine(bary0, betas=(40, 80, 150, 300), maxiter_per_stage=100):
    """
    Refine using beta annealing: start with smooth surrogate, increase sharpness.
    This helps navigate out of local minima of the hard objective.
    """
    bary = bary0.copy()
    for beta in betas:
        bary = _l_bfgs_softmin(bary, maxiter=maxiter_per_stage, beta=beta)
    return bary

def _differential_evolution_global(seed=42, maxiter=30, popsize=15):
    """Differential evolution for global search using soft-min objective."""
    try:
        bounds = [(0, 1)] * (2 * N)
        
        def de_obj(x):
            # Use a moderate beta for DE to keep the landscape smooth but informative
            return _softmin_objective(x, beta=60.0)
        
        de_result = differential_evolution(
            de_obj,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            updating='deferred'
        )
        
        bary_opt = _clamp_bary(de_result.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    def score_bary(bary):
        points = _bary_to_cart(bary)
        return _min_triangle_area_vectorized(points)
    
    def update_best(bary):
        nonlocal best_bary, best_score
        s = score_bary(bary)
        if s > best_score:
            best_score = s
            best_bary = bary.copy()
    
    # Phase 1: Global Search via Differential Evolution
    # We run a few DE instances with different seeds to explore the space broadly.
    de_seeds = [42, 123, 456]
    
    for seed in de_seeds:
        if time_left() < 10.0:
            break
        
        de_bary = _differential_evolution_global(seed=seed, maxiter=40, popsize=15)
        if de_bary is not None:
            # Refine the DE result with Beta Annealing
            if time_left() > 5.0:
                refined_bary = _beta_anneal_refine(de_bary, betas=(50, 100, 200), maxiter_per_stage=80)
                update_best(refined_bary)
            update_best(de_bary)
    
    # Phase 2: Multi-start Local Search with Structured Initializations
    # Use diverse structured starts and refine with Beta Annealing + Hard Polish
    initial_seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for i, seed in enumerate(initial_seeds):
        if time_left() < 5.0:
            break
        
        mode = i % 5
        bary_init = _generate_initial_bary(seed=seed, mode=mode)
        
        # Refine with Beta Annealing
        if time_left() > 3.0:
            refined_bary = _beta_anneal_refine(bary_init, betas=(40, 80, 150), maxiter_per_stage=60)
            update_best(refined_bary)
            
            # Final hard polish with Nelder-Mead or L-BFGS-B on hard objective
            if time_left() > 2.0:
                # Try Nelder-Mead as it doesn't rely on gradients which might be noisy for hard min
                nm_bary = _nelder_mead_optimize(refined_bary, maxiter=100)
                update_best(nm_bary)
                
    # Phase 3: Perturbation and Fine-tuning of the Best Solution
    # If we have a good candidate, try to improve it with small perturbations
    if best_bary is not None:
        rng = np.random.RandomState(999)
        for _ in range(10):
            if time_left() < 3.0:
                break
            
            # Perturb the best solution
            pert = rng.normal(0, 0.02, size=(N, 2))
            bary_start = _clamp_bary(best_bary + pert)
            
            # Quick Beta Anneal
            if time_left() > 2.0:
                refined_bary = _beta_anneal_refine(bary_start, betas=(100, 200), maxiter_per_stage=40)
                update_best(refined_bary)
                
                # Hard Polish
                if time_left() > 1.5:
                    nm_bary = _nelder_mead_optimize(refined_bary, maxiter=50)
                    update_best(nm_bary)

    # Final fallback
    if best_bary is None:
        best_bary = _generate_initial_bary(seed=42, mode=0)
        
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
