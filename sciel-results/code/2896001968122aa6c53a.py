"""Rewritten with minimax log-sum-exp + differential evolution + multi-start."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]
_NTRIP = len(_IDX)


def _clamp_one(x, y):
    """Clamp a single point to the triangle."""
    y = min(max(y, _EPS), _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = min(max(x, x_min), x_max)
    return x, y


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


def _logsumexp_neg(pts, beta):
    """
    Smooth approximation of -min(areas) using log-sum-exp.
    As beta -> inf, this approaches -min(areas).
    Returns a scalar.
    """
    areas = _areas_vec(pts)
    # We want to approximate -min(areas) = -(-max(-areas)) 
    # log-sum-exp of -areas with temperature 1/beta:
    # (1/beta) * log(sum(exp(-beta * areas)))  ->  min(areas) as beta->inf
    # So -min(areas) ~ -(1/beta)*log(sum(exp(-beta*areas)))
    # = (1/beta)*log(sum(exp(beta*areas)))  ... wait let me think again.
    #
    # min(areas) = -max(-areas)
    # max(-areas) ≈ (1/beta)*log(sum(exp(beta*(-areas))))  (soft max)
    # So min(areas) ≈ -(1/beta)*log(sum(exp(-beta*areas)))
    # And -min(areas) ≈ (1/beta)*log(sum(exp(-beta*areas)))
    
    neg_beta_areas = -beta * areas
    # Numerical stability
    m = np.max(neg_beta_areas)
    if m > 500:
        # overflow risk, scale
        neg_beta_areas = neg_beta_areas - m
        m = 0.0
    s = np.sum(np.exp(neg_beta_areas))
    if s < 1e-300:
        return 1e30
    return (m + np.log(s)) / beta


def _objective_lse(params, beta):
    """Smooth objective: log-sum-exp approximation of -min_area."""
    pts = _clamp_all(params.reshape(N, 2))
    return _logsumexp_neg(pts, beta)


def _objective_exact(params):
    """Exact non-smooth objective for DE (which handles non-smooth fine)."""
    pts = _clamp_all(params.reshape(N, 2))
    return -_min_area(pts)


def _gradient_refine(pts, beta_start=50.0, beta_end=2000.0, n_steps=30, time_limit=10.0):
    """
    Anneal beta from low to high while doing gradient steps.
    This progressively sharpens the objective from smooth to near-exact.
    """
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    best_x = x0.copy()
    best_score = _min_area(best_x.reshape(N, 2))
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        # Anneal beta
        frac = step / max(1, n_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        # Use L-BFGS-B on the smooth objective
        try:
            res = minimize(
                _objective_lse,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000},
            )
            cand = _clamp_all(res.x.reshape(N, 2))
            s = _min_area(cand)
            if s > best_score:
                best_score = s
                best_x = cand.flatten()
                x0 = best_x.copy()
        except Exception:
            break
    
    return _clamp_all(best_x.reshape(N, 2)), best_score


def _differential_evolution_optimize(pts, time_limit=15.0, seed=42):
    """
    Use scipy's differential_evolution for global search.
    Bounds: x in [0,1], y in [0, H]. The clamp handles the triangle constraint.
    """
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    
    # Bounds for DE: each coordinate
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, _H))   # y
    
    # Wrap the objective to enforce triangle constraint
    def _de_obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = _clamp_all(x.reshape(N, 2))
        return -_min_area(pts)
    
    try:
        res = differential_evolution(
            _de_obj,
            bounds=bounds,
            seed=seed,
            maxiter=200,
            popsize=15,
            tol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,  # We do our own polish
            init='sohal',
            update_interval=10,
        )
        if time.time() - start_time > time_limit:
            return _clamp_all(x0.reshape(N, 2)), _min_area(_clamp_all(x0.reshape(N, 2)))
        cand = _clamp_all(res.x.reshape(N, 2))
        s = _min_area(cand)
        return cand, s
    except Exception:
        return _clamp_all(x0.reshape(N, 2)), _min_area(_clamp_all(x0.reshape(N, 2)))


def _nelder_mead_refine(pts, time_limit=5.0):
    """Nelder-Mead as a secondary refiner."""
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    best_pts = x0.reshape(N, 2).copy()
    best_score = _min_area(best_pts)
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'maxfev': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                     'adaptive': True},
        )
        if time.time() - start_time < time_limit:
            cand = _clamp_all(res.x.reshape(N, 2))
            s = _min_area(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
    except Exception:
        pass
    return best_pts, best_score


def _generate_seeds():
    """Generate diverse initial configurations."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + 7 points on a hexagonal-like pattern
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
    
    # Seed 2: 3-4-4 triangular grid
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
    
    # Seed 3: Symmetric with 3-fold rotation
    # 3 vertices, 1 center, 3 at 1/3 height, 3 at 2/3 height
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
    
    # Seed 4: Random with fixed seed
    rng = np.random.default_rng(123)
    for _ in range(6):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        seeds.append(_clamp_all(pts))
    
    # Seed 5: Points near edges and center
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


def _perturb(pts, rng, scale=0.1):
    """Add Gaussian perturbation and clamp."""
    p = np.array(pts, dtype=float)
    p += rng.normal(0, scale, p.shape)
    return _clamp_all(p)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
    
    seeds = _generate_seeds()
    
    # Phase 1: Quick Nelder-Mead on each seed to get decent local optima
    n_seeds = len(seeds)
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        remaining = deadline - 30 - time.time()
        per = min(3.0, remaining / max(1, n_seeds - i))
        if per < 0.5:
            break
        try:
            opt_pts, score = _nelder_mead_refine(seed, time_limit=per)
            _update(opt_pts, score)
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = seeds[0]
        best_score = _min_area(best_pts)
    
    # Phase 2: Differential evolution from best + perturbed bests
    # DE is good for global exploration in the 22D space
    if time.time() < deadline - 40:
        remaining = deadline - 40 - time.time()
        if remaining > 5:
            # Run DE from the best point
            try:
                de_pts, de_score = _differential_evolution_optimize(
                    best_pts, time_limit=min(20.0, remaining), seed=7
                )
                _update(de_pts, de_score)
            except Exception:
                pass
    
    # Phase 3: Gradient refinement with annealed log-sum-exp
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - 15 - time.time()
        if remaining > 3:
            try:
                grad_pts, grad_score = _gradient_refine(
                    best_pts, beta_start=30, beta_end=3000,
                    n_steps=25, time_limit=min(12.0, remaining)
                )
                _update(grad_pts, grad_score)
            except Exception:
                pass
    
    # Phase 4: Another round of Nelder-Mead on the best (with adaptive)
    if best_pts is not None and time.time() < deadline - 8:
        remaining = deadline - 8 - time.time()
        if remaining > 2:
            try:
                nm_pts, nm_score = _nelder_mead_refine(
                    best_pts, time_limit=min(6.0, remaining)
                )
                _update(nm_pts, nm_score)
            except Exception:
                pass
    
    # Phase 5: Final gradient polish with very high beta (near-exact)
    if best_pts is not None and time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 1.5:
            try:
                final_pts, final_score = _gradient_refine(
                    best_pts, beta_start=500, beta_end=10000,
                    n_steps=15, time_limit=min(5.0, remaining)
                )
                _update(final_pts, final_score)
            except Exception:
                pass
    
    # Phase 6: One more DE run if time allows (different seed for diversity)
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 5:
            try:
                de_pts2, de_score2 = _differential_evolution_optimize(
                    _perturb(best_pts, np.random.default_rng(99), 0.05),
                    time_limit=min(10.0, remaining), seed=2024
                )
                _update(de_pts2, de_score2)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = seeds[0]
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
