"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Combined strategy:
- Barycentric coordinate representation for robust constraint handling
- Multi-stage optimization: SA for global exploration, L-BFGS-B with log-sum-exp smoothing, NM polish
- Vectorized area computation in barycentric coordinates
- Multi-start SLSQP for escaping local minima
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

# Barycentric vertices: A=(0,0), B=(1,0), C=(0.5, H)
# Point = u*A + v*B + w*C where w = 1-u-v
# x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
# y = w*H = (1-u-v)*H


def _cart_to_bary(pts):
    """Convert Cartesian (x,y) to barycentric (u,v) where w=1-u-v."""
    x = pts[:, 0]
    y = pts[:, 1]
    w = y / _H
    u = 1.0 - w - x
    v = x - 0.5 * w
    return np.column_stack([u, v])


def _bary_to_cart(bary):
    """Convert barycentric (u,v) to Cartesian (x,y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * _H
    return np.column_stack([x, y])


def _clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to valid region: u>=eps, v>=eps, u+v<=1-eps."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])


def _clamp_cart(pts):
    """Vectorized clamp to triangle in Cartesian coords."""
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


def _areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios (relative to big triangle area).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]
    vi = v[_I]
    wi = w[_I]
    
    uj = u[_J]
    vj = v[_J]
    wj = w[_J]
    
    uk = u[_K]
    vk = v[_K]
    wk = w[_K]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)


def _min_area_bary(bary):
    """Minimum triangle area for barycentric points."""
    return float(np.min(_areas_bary(bary)))


def _min_area_cart(pts):
    """Minimum triangle area for Cartesian points."""
    bary = _cart_to_bary(pts)
    return _min_area_bary(bary)


def _obj_hard(bary_flat):
    """Exact non-smooth objective: negative min area."""
    bary = _clamp_bary(bary_flat.reshape(N, 2))
    return -_min_area_bary(bary)


def _obj_soft(bary_flat, beta=100.0):
    """
    Objective function using soft min (log-sum-exp). Returns negative soft-min area.
    """
    bary = _clamp_bary(bary_flat.reshape(N, 2))
    areas = _areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min


def _sa_global(bary_init, time_limit=30.0, seed=42):
    """
    Simulated annealing for global exploration in barycentric space.
    """
    rng = np.random.default_rng(seed)
    bary = _clamp_bary(np.array(bary_init, dtype=float).copy())
    best_bary = bary.copy()
    best_score = _min_area_bary(bary)
    
    t0 = time.time()
    T = 0.05  # initial temperature
    cooling = 0.995
    
    while time.time() - t0 < time_limit:
        T *= cooling
        if T < 1e-6:
            T = 1e-6
        
        # Generate perturbation
        noise_scale = 0.02 + 0.03 * T / 0.05
        new_bary = bary + rng.normal(0, noise_scale, bary.shape)
        new_bary = _clamp_bary(new_bary)
        
        new_score = _min_area_bary(new_bary)
        delta = new_score - _min_area_bary(bary)
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            bary = new_bary
            if new_score > best_score:
                best_score = new_score
                best_bary = bary.copy()
    
    return _clamp_bary(best_bary), best_score


def _lbfgs_refine(bary, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, time_limit=20.0):
    """
    Multi-stage L-BFGS-B refinement with log-sum-exp smoothing.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    best_bary = _clamp_bary(np.array(bary, dtype=float).copy())
    best_score = _min_area_bary(best_bary)
    
    # Stage 1: Exploration with smooth landscape
    try:
        result1 = minimize(
            _obj_soft,
            best_bary.flatten(),
            args=(beta1,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        cand1 = _clamp_bary(result1.x.reshape(N, 2))
        s1 = _min_area_bary(cand1)
        if s1 > best_score:
            best_score = s1
            best_bary = cand1
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.6:
        return best_bary, best_score
    
    # Stage 2: Exploitation with sharper landscape
    try:
        result2 = minimize(
            _obj_soft,
            best_bary.flatten(),
            args=(beta2,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        cand2 = _clamp_bary(result2.x.reshape(N, 2))
        s2 = _min_area_bary(cand2)
        if s2 > best_score:
            best_score = s2
            best_bary = cand2
    except Exception:
        pass
    
    return best_bary, best_score


def _slsqp_refine(bary, time_limit=15.0, maxiter=500):
    """
    SLSQP refinement on the hard min objective in barycentric space.
    SLSQP handles constraints more naturally and can escape local minima better.
    """
    start_time = time.time()
    best_bary = _clamp_bary(np.array(bary, dtype=float).copy())
    best_score = _min_area_bary(best_bary)
    
    # Define constraints: u >= eps, v >= eps, u+v <= 1-eps for each point
    def constraint_u(x):
        return x[0::2] - _EPS  # u_i >= eps
    
    def constraint_v(x):
        return x[1::2] - _EPS  # v_i >= eps
    
    def constraint_sum(x):
        return (1.0 - _EPS) - (x[0::2] + x[1::2])  # u_i + v_i <= 1-eps
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_u},
        {'type': 'ineq', 'fun': constraint_v},
        {'type': 'ineq', 'fun': constraint_sum}
    ]
    
    bounds = [(0, 1)] * (N * 2)
    
    try:
        res = minimize(
            _obj_hard,
            best_bary.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        cand = _clamp_bary(res.x.reshape(N, 2))
        s = _min_area_bary(cand)
        if s > best_score:
            best_score = s
            best_bary = cand
    except Exception:
        pass
    
    return best_bary, best_score


def _nm_polish(bary, time_limit=10.0, maxiter=500):
    """
    Nelder-Mead polish on the hard min objective in barycentric space.
    """
    start_time = time.time()
    x0 = _clamp_bary(np.array(bary, dtype=float)).flatten()
    
    best_bary = x0.copy().reshape(N, 2)
    best_score = _min_area_bary(best_bary)
    
    try:
        res = minimize(
            _obj_hard,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            },
        )
        cand = _clamp_bary(res.x.reshape(N, 2))
        s = _min_area_bary(cand)
        if s > best_score:
            best_score = s
            best_bary = cand
    except Exception:
        pass
    
    # Second NM pass if time remains
    if time.time() - start_time < time_limit * 0.6:
        try:
            res2 = minimize(
                _obj_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': maxiter // 2,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True,
                },
            )
            cand2 = _clamp_bary(res2.x.reshape(N, 2))
            s2 = _min_area_bary(cand2)
            if s2 > best_score:
                best_score = s2
                best_bary = cand2
        except Exception:
            pass
    
    return _clamp_bary(best_bary), best_score


def _generate_seeds_bary():
    """Generate diverse initial configurations in barycentric coordinates."""
    seeds = []
    
    # Seed 1: Vertices + centroid + structured points
    s1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1/3, 1/3],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.35, 0.45],
        [0.65, 0.45],
    ])
    seeds.append(_clamp_bary(s1))
    
    # Seed 2: Triangular grid pattern
    s2 = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.5],
        [0.75, 0.5],
        [0.5, 0.75],
        [0.25, 0.75],
    ])
    seeds.append(_clamp_bary(s2))
    
    # Seed 3: 3-fold symmetric
    s3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1/3, 1/3],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.5, 0.5],
        [0.35, 0.45],
        [0.65, 0.45],
        [0.25, 0.25],
        [0.75, 0.25],
    ])
    seeds.append(_clamp_bary(s3))
    
    # Seed 4: Random seeds
    rng = np.random.default_rng(123)
    for _ in range(8):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        seeds.append(_clamp_bary(np.column_stack([u, v])))
        
    # Seed 5: Near edges
    s5 = np.array([
        [0.1, 0.05],
        [0.9, 0.05],
        [0.05, 0.9],
        [0.35, 0.35],
        [0.25, 0.2],
        [0.7, 0.2],
        [0.5, 0.5],
        [0.15, 0.3],
        [0.8, 0.3],
        [0.4, 0.6],
        [0.6, 0.6],
    ])
    seeds.append(_clamp_bary(s5))
    
    # Seed 6: Hexagonal lattice approximation
    s6 = np.array([
        [0.1, 0.1],
        [0.4, 0.1],
        [0.7, 0.1],
        [0.25, 0.35],
        [0.55, 0.35],
        [0.15, 0.6],
        [0.45, 0.6],
        [0.75, 0.6],
        [0.3, 0.8],
        [0.6, 0.8],
        [0.45, 0.45],
    ])
    seeds.append(_clamp_bary(s6))
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -1.0
    
    def _update(bary, score):
        nonlocal best_bary, best_score
        if score > best_score:
            best_score = score
            best_bary = _clamp_bary(np.array(bary, dtype=float).copy())
    
    # Generate seeds
    seeds = _generate_seeds_bary()
    
    # Phase 1: SA global exploration from best seed
    # First, evaluate all seeds and pick the best
    for seed in seeds:
        s = _min_area_bary(seed)
        if s > best_score:
            best_score = s
            best_bary = seed.copy()
    
    if best_bary is None:
        best_bary = seeds[0]
        best_score = _min_area_bary(best_bary)
    
    # Phase 2: SA from top seeds
    if time.time() < deadline - 60:
        remaining = deadline - 60 - time.time()
        if remaining > 15:
            # Sort seeds by score
            seed_scores = [(seed, _min_area_bary(seed)) for seed in seeds]
            seed_scores.sort(key=lambda x: x[1], reverse=True)
            
            sa_time = min(25.0, remaining * 0.5)
            for i, (seed, _) in enumerate(seed_scores[:3]):
                if time.time() > deadline - 60:
                    break
                per = min(sa_time / max(1, 3 - i), remaining - (time.time() - (deadline - 60)))
                if per < 3:
                    break
                try:
                    sa_bary, sa_score = _sa_global(seed, time_limit=per, seed=42 + i)
                    _update(sa_bary, sa_score)
                except Exception:
                    continue
    
    # Phase 3: L-BFGS-B refinement from best and top SA results
    if time.time() < deadline - 30:
        remaining = deadline - 30 - time.time()
        if remaining > 10:
            # Refine current best
            try:
                lbfgs_bary, lbfgs_score = _lbfgs_refine(
                    best_bary, beta1=50.0, beta2=200.0,
                    maxiter1=200, maxiter2=300,
                    time_limit=min(20.0, remaining * 0.4)
                )
                _update(lbfgs_bary, lbfgs_score)
            except Exception:
                pass
            
            # Try a few perturbed starts
            rng = np.random.default_rng(777)
            for i in range(3):
                if time.time() > deadline - 25:
                    break
                perturbed = best_bary + rng.normal(0, 0.03, best_bary.shape)
                perturbed = _clamp_bary(perturbed)
                try:
                    lbfgs_bary2, lbfgs_score2 = _lbfgs_refine(
                        perturbed, beta1=80.0, beta2=250.0,
                        maxiter1=150, maxiter2=200,
                        time_limit=min(10.0, remaining * 0.2)
                    )
                    _update(lbfgs_bary2, lbfgs_score2)
                except Exception:
                    continue
    
    # Phase 4: SLSQP multi-start refinement (key improvement from inspiration)
    if time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        if remaining > 8:
            # Try SLSQP from multiple starting points to escape local minima
            rng = np.random.default_rng(999)
            start_points = [best_bary.copy()]
            
            # Add perturbed versions
            for _ in range(4):
                perturbed = best_bary + rng.normal(0, 0.05, best_bary.shape)
                start_points.append(_clamp_bary(perturbed))
            
            # Also add some random starts
            for _ in range(2):
                u = rng.random(N)
                v = rng.random(N) * (1 - u)
                start_points.append(_clamp_bary(np.column_stack([u, v])))
            
            for i, sp in enumerate(start_points):
                if time.time() > deadline - 15:
                    break
                per = min(12.0, remaining * 0.2)
                try:
                    slsqp_bary, slsqp_score = _slsqp_refine(
                        sp, time_limit=per, maxiter=300
                    )
                    _update(slsqp_bary, slsqp_score)
                except Exception:
                    continue
    
    # Phase 5: NM polish
    if best_bary is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            try:
                nm_bary, nm_score = _nm_polish(
                    best_bary, time_limit=min(10.0, remaining * 0.6), maxiter=500
                )
                _update(nm_bary, nm_score)
            except Exception:
                pass
    
    # Phase 6: Final tight NM
    if best_bary is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 3:
            try:
                nm2_bary, nm2_score = _nm_polish(
                    best_bary, time_limit=min(5.0, remaining * 0.7), maxiter=300
                )
                _update(nm2_bary, nm2_score)
            except Exception:
                pass
    
    # Fallback
    if best_bary is None:
        best_bary = seeds[0]
    
    # Convert to Cartesian and final safety clamp
    result_cart = _bary_to_cart(_clamp_bary(best_bary))
    result_cart = _clamp_cart(result_cart)
    
    return result_cart

# EVOLVE-BLOCK-END
