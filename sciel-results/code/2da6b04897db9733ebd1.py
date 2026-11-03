"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _bary_to_cart(bary):
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _softmin_objective(x, beta=120.0):
    """Smooth surrogate: negative soft-min of triangle areas using log-sum-exp.
    Higher beta gives sharper approximation to true min."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Numerically stable log-sum-exp for soft-min
    min_area = np.min(areas)
    shifted = areas - min_area
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

def _structured_init_bary(seed=42):
    """Generate structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric with vertices, edge midpoints, centroid
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
    
    # Pattern C: More spread
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Random structured
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
    
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _random_init_bary(seed):
    """Random initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary[i] = [u, v]
                break
    return bary

def _l_bfgs_softmin(bary0, maxiter=400, beta=120.0):
    """L-BFGS-B optimization using soft-min surrogate for smooth gradients."""
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

def _l_bfgs_hard(bary0, maxiter=200):
    """L-BFGS-B optimization using hard min (for final polish)."""
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

def _active_set_refine(bary0, maxiter=150):
    """Refine by identifying active (binding) triples and equalizing them via penalty.
    
    Uses a two-phase penalty: first push all areas above target, then equalize
    the binding set to maximize the minimum.
    """
    try:
        points = _bary_to_cart(bary0)
        areas = _all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Identify active triples (within tolerance of minimum)
        tol = max(0.03 * min_area, 1e-8)
        active_mask = areas <= min_area + tol
        active_count = np.sum(active_mask)
        
        if active_count < 2:
            return bary0.copy()
        
        target = min_area
        
        def penalty_obj(x):
            bary = x.reshape(N, 2)
            bary = _clamp_bary(bary)
            pts = _bary_to_cart(bary)
            a = _all_triangle_areas(pts)
            min_a = np.min(a)
            # Strong penalty for any area below target
            below_target = np.maximum(0, target - a)
            penalty_strong = 500.0 * np.sum(below_target ** 2)
            # Moderate penalty to equalize active set around target
            active_dev = a[active_mask] - target
            penalty_equalize = 50.0 * np.sum(active_dev ** 2)
            return -min_a + penalty_strong + penalty_equalize
        
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            penalty_obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        
        # Verify improvement
        score_opt = _score_bary(bary_opt)
        score_orig = _score_bary(bary0)
        if score_opt > score_orig:
            return bary_opt
        return bary0.copy()
    except Exception:
        return bary0.copy()

def _differential_evolution_global(seed=42, maxiter=40, popsize=14):
    """Differential evolution for global search using soft-min objective."""
    try:
        bounds = [(0, 1)] * (2 * N)
        
        def de_obj(x):
            return _softmin_objective(x, beta=80.0)
        
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

def _nelder_mead_optimize(bary0, maxiter=200):
    """Nelder-Mead for local refinement with tighter tolerances."""
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

def _polish_high_precision(bary0, maxiter=300):
    """High-precision polish using very high beta soft-min then hard min."""
    try:
        # Phase 1: High-beta soft-min for near-exact gradients
        bary1 = _l_bfgs_softmin(bary0, maxiter=maxiter, beta=200.0)
        score1 = _score_bary(bary1)
        
        # Phase 2: Hard min polish
        bary2 = _l_bfgs_hard(bary1, maxiter=maxiter // 2)
        score2 = _score_bary(bary2)
        
        if score2 >= score1:
            return bary2
        return bary1
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Multi-start with soft-min L-BFGS-B (smooth gradients)
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in initial_seeds:
        if time_left() < 5.0:
            break
        
        bary_init = _structured_init_bary(seed=seed)
        score_init = _score_bary(bary_init)
        
        # Soft-min L-BFGS-B: smooth gradients guide optimization
        if time_left() > 4.0:
            bary_opt = _l_bfgs_softmin(bary_init, maxiter=350, beta=120.0)
            score_opt = _score_bary(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        # Hard L-BFGS-B polish
        if time_left() > 3.0:
            bary_polish = _l_bfgs_hard(bary_init, maxiter=200)
            score_polish = _score_bary(bary_polish)
            if score_polish > score_init:
                score_init = score_polish
                bary_init = bary_polish
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 2: Differential evolution for global search
    if time_left() > 15.0:
        de_bary = _differential_evolution_global(seed=42, maxiter=35, popsize=12)
        if de_bary is not None:
            score_de = _score_bary(de_bary)
            if score_de > best_score:
                best_score = score_de
                best_bary = de_bary.copy()
                
                # Refine DE result with high-precision polish
                if time_left() > 8.0:
                    bary_ref = _polish_high_precision(de_bary, maxiter=250)
                    score_ref = _score_bary(bary_ref)
                    if score_ref > best_score:
                        best_score = score_ref
                        best_bary = bary_ref.copy()
    
    # Phase 3: Multi-start with random initializations
    rng = np.random.RandomState(123)
    for rs in range(8):
        if time_left() < 5.0:
            break
        
        bary_rand = _random_init_bary(seed=1000 + rs)
        score_rand = _score_bary(bary_rand)
        
        if time_left() > 4.0:
            bary_opt = _l_bfgs_softmin(bary_rand, maxiter=300, beta=120.0)
            score_opt = _score_bary(bary_opt)
            if score_opt > score_rand:
                score_rand = score_opt
                bary_rand = bary_opt
        
        if time_left() > 3.0:
            bary_polish = _l_bfgs_hard(bary_rand, maxiter=200)
            score_polish = _score_bary(bary_polish)
            if score_polish > score_rand:
                score_rand = score_polish
                bary_rand = bary_polish
        
        if score_rand > best_score:
            best_score = score_rand
            best_bary = bary_rand.copy()
    
    # Phase 4: Active-set refinement on best solution
    if time_left() > 5.0:
        bary_as = _active_set_refine(best_bary, maxiter=150)
        score_as = _score_bary(bary_as)
        if score_as > best_score:
            best_score = score_as
            best_bary = bary_as.copy()
    
    # Phase 5: Multi-start perturbations around best
    restart_count = 0
    max_restarts = 50
    while restart_count < max_restarts and time_left() > 4.0:
        restart_count += 1
        
        # Perturb the best solution with adaptive scale
        scale = 0.03 * (1.0 - 0.5 * restart_count / max_restarts) + 0.01
        pert = rng.normal(0, scale, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        # Soft-min optimization
        if time_left() > 3.5:
            bary_opt = _l_bfgs_softmin(bary_start, maxiter=250, beta=120.0)
            score_opt = _score_bary(bary_opt)
            
            # Hard polish
            if score_opt > best_score and time_left() > 2.5:
                bary_polish = _l_bfgs_hard(bary_opt, maxiter=150)
                score_polish = _score_bary(bary_polish)
                if score_polish > score_opt:
                    score_opt = score_polish
                    bary_opt = bary_polish
            
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Phase 6: Nelder-Mead fine-tuning
    if time_left() > 5.0:
        bary_nm = _nelder_mead_optimize(best_bary, maxiter=150)
        score_nm = _score_bary(bary_nm)
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
    
    # Phase 7: High-precision final polish
    if time_left() > 6.0:
        bary_hp = _polish_high_precision(best_bary, maxiter=300)
        score_hp = _score_bary(bary_hp)
        if score_hp > best_score:
            best_score = score_hp
            best_bary = bary_hp.copy()
    
    # Phase 8: Final active-set + high-precision combo
    if time_left() > 4.0:
        bary_as = _active_set_refine(best_bary, maxiter=100)
        score_as = _score_bary(bary_as)
        if score_as > best_score:
            best_score = score_as
            best_bary = bary_as.copy()
        
        if time_left() > 3.0:
            bary_hp = _polish_high_precision(best_bary, maxiter=200)
            score_hp = _score_bary(bary_hp)
            if score_hp > best_score:
                best_score = score_hp
                best_bary = bary_hp.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
