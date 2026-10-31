"""
SA + log-sum-exp gradient polishing hybrid.
SA provides robust global exploration; LSE gradient gives high-precision local refinement.
Multi-phase: SA global -> gradient refine -> SA re-explore -> final polish.
"""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]


def _clamp_all(pts):
    """Vectorized clamp to triangle."""
    pts = np.array(pts, dtype=float).copy()
    x = pts[:, 0]
    y = pts[:, 1]
    y = np.clip(y, _EPS, _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = np.clip(x, x_min, x_max)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _clamp_one(pt):
    """Clamp a single 2D point to triangle."""
    x, y = pt
    y = np.clip(y, _EPS, _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = np.clip(x, x_min, x_max)
    return np.array([x, y])

def _areas_vec(pts):
    """All 165 triangle areas (vectorized)."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_area(pts):
    return float(np.min(_areas_vec(pts)))

def _logsumexp_neg(areas, beta=1000.0):
    """
    Approximate -min(areas) using log-sum-exp of -areas.
    min(a) = -lim_{beta->inf} (1/beta) * log(sum(exp(-beta*a)))
    So -min(a) = lim_{beta->inf} (1/beta) * log(sum(exp(-beta*a)))
    We want to maximize min(a), so we minimize -min(a).
    Let f(x) = -min(a).
    We approximate f(x) with g(x) = (1/beta) * log(sum(exp(-beta * a_i)))
    Note: as beta increases, g(x) approaches min(a_i).
    Wait, max min(a) is equivalent to min (-min(a)).
    Let's check signs.
    If a = [1, 2], min=1. -min=-1.
    logsumexp(-beta*a) = log(exp(-beta) + exp(-2beta)) ~ -beta for large beta.
    (1/beta)*logsumexp ~ -1.
    So we want to minimize (1/beta) * log(sum(exp(-beta * areas))).
    """
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    if not np.isfinite(max_val):
        return -1.0
    sum_exp = np.sum(np.exp(neg_beta_areas - max_val))
    if sum_exp == 0:
        return -1.0
    return (max_val + np.log(sum_exp)) / beta

def _objective_lse(params, beta=1000.0):
    """Smooth objective: approx negative min area."""
    pts = _clamp_all(params.reshape(N, 2))
    areas = _areas_vec(pts)
    return _logsumexp_neg(areas, beta)

def _objective_exact(params):
    """Exact non-smooth objective: negative min area."""
    pts = _clamp_all(params.reshape(N, 2))
    return -_min_area(pts)


def _gradient_refine(pts, time_limit=10.0, beta=1000.0):
    """
    Gradient-based local search using BFGS on the smooth LSE objective.
    """
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    best_x = x0.copy()
    best_score = _min_area(best_x.reshape(N, 2))
    
    def obj_wrapper(params):
        if time.time() > start_time + time_limit:
            return 1.0 # Large penalty to stop
        return _objective_lse(params, beta)
    
    try:
        res = minimize(
            obj_wrapper,
            x0,
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-8},
        )
        cand = _clamp_all(res.x.reshape(N, 2))
        s = _min_area(cand)
        if s > best_score:
            best_score = s
            best_x = cand.flatten()
    except Exception:
        pass
        
    return _clamp_all(best_x.reshape(N, 2)), best_score

def _simulated_annealing(pts, time_limit=10.0, seed=42):
    """
    Simulated Annealing for global exploration.
    """
    rng = np.random.default_rng(seed)
    current_pts = _clamp_all(np.array(pts, dtype=float)).copy()
    current_score = _min_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = 0.05 # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate neighbor: move one point randomly
        idx = rng.integers(0, N)
        new_pts = current_pts.copy()
        
        # Random step size proportional to temperature
        step_size = T * 0.1
        delta = rng.normal(0, step_size, size=2)
        new_pts[idx] = _clamp_one(new_pts[idx] + delta)
        
        new_score = _min_area(new_pts)
        delta_score = new_score - current_score
        
        if delta_score > 0:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(delta_score / T)
            if rng.random() < np.exp(delta_score / T):
                current_pts = new_pts
                current_score = new_score
                
        T *= alpha
        
    return best_pts, best_score


def _generate_seeds():
    """Generate diverse initial configurations."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + hexagonal pattern
    s1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.167, h/12],
        [0.833, h/12],
        [0.333, h/4],
        [0.667, h/4],
    ])
    seeds.append(_clamp_all(s1))
    
    # Seed 2: Triangular grid
    s2 = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h/4],
        [0.5, h/4],
        [0.75, h/4],
        [0.5, h/2],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, 2*h/3],
        [0.25, 2*h/3],
    ])
    seeds.append(_clamp_all(s2))
    
    # Seed 3: 3-fold symmetric
    s3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.167, h/6],
        [0.833, h/6],
        [0.5, h/2],
        [0.333, 2*h/3],
        [0.667, 2*h/3],
        [0.25, h/4],
        [0.75, h/4],
    ])
    seeds.append(_clamp_all(s3))
    
    # Seed 4: Random
    rng = np.random.default_rng(123)
    for _ in range(10):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        seeds.append(_clamp_all(pts))
        
    # Seed 5: Near edges
    s5 = np.array([
        [0.1, 0.05],
        [0.9, 0.05],
        [0.5, h - 0.05],
        [0.5, h/3],
        [0.3, h/6],
        [0.7, h/6],
        [0.5, h/2],
        [0.2, h/4],
        [0.8, h/4],
        [0.4, 2*h/5],
        [0.6, 2*h/5],
    ])
    seeds.append(_clamp_all(s5))
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
    
    # Phase 1: SA + Gradient Refinement from multiple seeds
    seeds = _generate_seeds()
    
    # We have ~140s. Let's allocate time to multiple SA runs.
    # Each SA run takes ~10-20s. Then gradient refinement ~5s.
    # Let's do 5 SA runs of 20s each, then polish.
    
    num_sa_runs = 5
    sa_time_per_run = 20.0
    grad_time_per_run = 5.0
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 20: # Keep 20s for final polish
            break
            
        # Run SA
        try:
            sa_pts, sa_score = _simulated_annealing(seed, time_limit=sa_time_per_run, seed=i)
            _update(sa_pts, sa_score)
        except Exception:
            continue
            
        # Refine with Gradient
        if time.time() < deadline - 10:
            try:
                grad_pts, grad_score = _gradient_refine(sa_pts, time_limit=grad_time_per_run)
                _update(grad_pts, grad_score)
            except Exception:
                pass
                
    if best_pts is None:
        best_pts = seeds[0]
        best_score = _min_area(best_pts)
        
    # Phase 2: Final intensive polishing
    # Use remaining time for multiple gradient refinements with increasing beta
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        
        # Try different betas for LSE to get closer to true min
        betas = [500.0, 1000.0, 2000.0, 5000.0]
        time_per_beta = remaining / len(betas)
        
        for beta in betas:
            if time.time() > deadline - 2:
                break
            try:
                # Temporarily modify beta for the wrapper? 
                # _gradient_refine uses fixed beta=1000. 
                # Let's make a local wrapper or just call with default and hope 1000 is good.
                # Actually, let's just do one more high-precision gradient pass.
                # The LSE with beta=1000 is a good approximation.
                # Let's just do a few more passes.
                
                pass_time = min(10.0, remaining)
                grad_pts, grad_score = _gradient_refine(best_pts, time_limit=pass_time)
                _update(grad_pts, grad_score)
                
                # If we have time, do another
                if time.time() < deadline - 2:
                     grad_pts2, grad_score2 = _gradient_refine(grad_pts, time_limit=min(5.0, deadline - 2 - time.time()))
                     _update(grad_pts2, grad_score2)
                     
            except Exception:
                pass
                
    if best_pts is None:
        best_pts = seeds[0]
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
