"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

_start = time.time()
DEADLINE = 150.0
SAFETY = 10.0

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian. bary shape (N, 3)."""
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian to barycentric coordinates. pts shape (N, 2)."""
    x = pts[:, 0]
    y = pts[:, 1]
    # bary[1] = y / h where h = sqrt(3)/2
    h = np.sqrt(3) / 2
    b1 = y / h
    b2 = x - b1 * V2[0]
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to valid range [0,1] with sum=1."""
    bary = np.clip(bary, eps, None)
    bary /= bary.sum(axis=1, keepdims=True)
    return bary


def _min_triangle_area_vectorized(pts):
    """Compute minimum triangle area from Cartesian points."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _all_triangle_areas(pts):
    """Compute all triangle areas."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _softmin_objective(x, beta=100.0):
    """Soft-min surrogate for the minimum triangle area. x is flat Cartesian coords."""
    pts = x.reshape(N, 2)
    areas = _all_triangle_areas(pts)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This is differentiable and approximates -min(areas) for large beta
    max_a = areas.max()
    exp_terms = np.exp(-beta * (areas - max_a))
    soft_min = -max_a - (1.0 / beta) * np.log(np.sum(exp_terms))
    return soft_min


def _hard_objective(x):
    """Hard minimum triangle area (negative for minimization)."""
    pts = x.reshape(N, 2)
    areas = _all_triangle_areas(pts)
    return -areas.min()


def _score_bary(bary):
    """Score a barycentric configuration."""
    pts = _bary_to_cart(bary)
    return _min_triangle_area_vectorized(pts)


def _structured_init_bary(rng):
    """Generate a structured initial configuration in barycentric coords."""
    # Use a grid-like pattern in barycentric space
    bary = np.zeros((N, 3))
    
    # Center point
    bary[0] = [1/3, 1/3, 1/3]
    
    # 5 points in inner ring
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi/5
        r = 0.25
        # Approximate barycentric from polar around center
        # Center in barycentric is (1/3, 1/3, 1/3)
        # Perturb in barycentric space
        bary[1+i] = [1/3 + r*np.cos(angle)*0.5, 1/3 + r*np.sin(angle)*0.5, 1/3]
    
    # 5 points in outer ring
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.4
        bary[6+i] = [1/3 + r*np.cos(angle)*0.5, 1/3 + r*np.sin(angle)*0.5, 1/3]
    
    bary = _clamp_bary(bary)
    return bary


def _random_init_bary(rng):
    """Generate random valid barycentric configuration."""
    # Sample random points in triangle via barycentric
    bary = np.zeros((N, 3))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        bary[i] = [1 - u - v, u, v]
    return bary


def _l_bfgs_softmin(bary_init, beta_start=50.0, beta_end=500.0, n_iters=50, rng=None):
    """Optimize using L-BFGS-B with adaptive beta annealing."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    x0 = _bary_to_cart(bary_init).flatten()
    best_bary = bary_init.copy()
    best_score = _score_bary(bary_init)
    
    n_steps = n_iters
    for step in range(n_steps):
        if time.time() - _start > DEADLINE - SAFETY:
            break
        
        # Adaptive beta: start low, increase for sharper approximation
        frac = step / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        # Objective for this step
        def obj(x):
            return _softmin_objective(x, beta)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = result.x
            pts_opt = x_opt.reshape(N, 2)
            bary_opt = _cart_to_bary(pts_opt)
            bary_opt = _clamp_bary(bary_opt)
            
            score = _score_bary(bary_opt)
            if score > best_score:
                best_score = score
                best_bary = bary_opt.copy()
            
            x0 = bary_opt.flatten() if False else pts_opt.flatten()
        except Exception:
            pass
    
    return best_bary, best_score


def _slsqp_refine(bary_init, maxiter=300):
    """Refine using SLSQP with barycentric constraints."""
    x0 = bary_init.flatten()
    
    # Constraints: barycentric coords sum to 1 and are non-negative
    def constraint_sum(i):
        def c(x):
            return x[3*i:3*i+3].sum() - 1.0
        return c
    
    constraints = [
        {'type': 'eq', 'fun': constraint_sum(i), 'jac': None}
        for i in range(N)
    ]
    
    # Bounds: each barycentric coord in [0, 1]
    bounds = [(0, 1)] * (N * 3)
    
    def obj(x):
        bary = x.reshape(N, 3)
        bary = _clamp_bary(bary)
        pts = _bary_to_cart(bary)
        return _hard_objective(pts.flatten())
    
    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-12})
        bary_opt = _clamp_bary(result.x.reshape(N, 3))
        score = _score_bary(bary_opt)
        return bary_opt, score
    except Exception:
        return bary_init, _score_bary(bary_init)


def _differential_evolution_global(rng, pop_size=20, n_gen=30):
    """Differential evolution in barycentric space."""
    # Initialize population
    pop = [_random_init_bary(rng) for _ in range(pop_size)]
    pop[0] = _structured_init_bary(rng)
    scores = [_score_bary(b) for b in pop]
    best_idx = np.argmax(scores)
    best_bary = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(n_gen):
        if time.time() - _start > DEADLINE - SAFETY:
            break
        
        new_pop = [pop[best_idx].copy()]  # Elitism
        
        for i in range(1, pop_size):
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, 3, replace=False)
            F = 0.8  # Scaling factor
            # Differential evolution step in barycentric space
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = _clamp_bary(mutant)
            
            score = _score_bary(mutant)
            if score > scores[i]:
                new_pop.append(mutant)
                scores[i] = score
                if score > best_score:
                    best_score = score
                    best_bary = mutant.copy()
            else:
                new_pop.append(pop[i])
        
        pop = new_pop
    
    return best_bary, best_score


def construct():
    rng = np.random.RandomState(42)
    
    best_bary = None
    best_score = 0.0
    
    # Phase 1: Differential evolution for global search
    de_bary, de_score = _differential_evolution_global(rng, pop_size=15, n_gen=20)
    if de_score > best_score:
        best_score = de_score
        best_bary = de_bary
    
    # Phase 2: Multiple L-BFGS-B refinements with soft-min from different starts
    starts = [
        _structured_init_bary(rng),
        _random_init_bary(rng),
        _random_init_bary(rng),
        de_bary.copy() if de_bary is not None else _random_init_bary(rng),
    ]
    
    for i, start_bary in enumerate(starts):
        if time.time() - _start > DEADLINE - SAFETY:
            break
        
        bary_opt, score = _l_bfgs_softmin(start_bary, beta_start=30, beta_end=300, n_iters=40, rng=rng)
        if score > best_score:
            best_score = score
            best_bary = bary_opt
    
    # Phase 3: SLSQP refinement on best solution
    if best_bary is not None:
        bary_slsqp, score_slsqp = _slsqp_refine(best_bary, maxiter=200)
        if score_slsqp > best_score:
            best_score = score_slsqp
            best_bary = bary_slsqp
    
    # Final: one more round of L-BFGS-B from the best
    if best_bary is not None and time.time() - _start < DEADLINE - SAFETY:
        bary_final, score_final = _l_bfgs_softmin(best_bary, beta_start=100, beta_end=800, n_iters=30, rng=rng)
        if score_final > best_score:
            best_score = score_final
            best_bary = bary_final
    
    # Convert to Cartesian
    if best_bary is not None:
        pts = _bary_to_cart(best_bary)
    else:
        # Fallback
        pts = _bary_to_cart(_random_init_bary(rng))
    
    return pts
