"""
Heilbronn Triangles: Maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: Combined structural approach using:
1. Barycentric area computation (determinant-based) for speed and precision
2. Multi-start L-BFGS-B with annealed soft-min
3. Hexagonal lattice initialization
4. Nelder-Mead final polish on hard min
5. Perturbation-based local search
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area values.
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
    
    # Determinant calculation: area = 0.5 * |det|
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return 0.5 * np.abs(det)

def _clamp_bary(bary, eps=1e-8):
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])

def _bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*(0,0) + v*(1,0) + w*(0.5, H) where w = 1-u-v
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def _soft_min_objective(x, beta, penalty_weight):
    """
    Objective function for L-BFGS-B.
    x: flat array of length 2*N (u, v pairs).
    Returns: -soft_min_area + penalty. (Minimizing this maximizes area and satisfies constraints).
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Penalty for violating u >= 0, v >= 0, u+v <= 1
    viol_u = np.maximum(0.0, -u)
    viol_v = np.maximum(0.0, -v)
    viol_sum = np.maximum(0.0, u + v - 1.0)
    
    penalty = penalty_weight * (np.sum(viol_u**2) + np.sum(viol_v**2) + np.sum(viol_sum**2))
    
    # Clamp to valid region for area calculation
    uv_clamped = _clamp_bary(uv)
    
    areas = _calc_areas_bary(uv_clamped)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    if max_a < 1e-14:
        return 1e10 + penalty
    
    # Avoid overflow in exp
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min = max_a - log_sum / beta
    
    return -soft_min + penalty

def _hard_min_objective(x):
    """
    Objective function using hard min for final polishing.
    """
    uv = x.reshape(N, 2)
    uv_clamped = _clamp_bary(uv)
    areas = _calc_areas_bary(uv_clamped)
    return -np.min(areas)

def _generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * 0.5  # Simplified for barycentric
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        
        if len(points) >= N:
            bary = np.array(points[:N])
            bary = _clamp_bary(bary)
            areas = _calc_areas_bary(bary)
            val = np.min(areas)
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def _generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    return _clamp_bary(np.column_stack([u, v]))

def _generate_perturbed_init(base, noise_scale=0.03, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return _clamp_bary(bary)

def _generate_structured_init():
    """Generate structured initial configurations."""
    configs = []
    rng = np.random.default_rng(123)
    
    # 1. Random valid points
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        configs.append(_clamp_bary(np.column_stack([u, v])))
        
    # 2. Symmetric/Geometric points
    special = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1/3, 1/3),
        (0.2, 0.2),
        (0.6, 0.2),
        (0.2, 0.6),
        (0.5, 0.25)
    ]
    configs.append(_clamp_bary(np.array(special[:N])))
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    best_score = -1.0
    best_uv = None
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def _try_optimize(init_uv, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        Returns the optimized barycentric coordinates.
        """
        if time.time() > deadline - 10.0:
            return init_uv
        
        try:
            # Stage 1: Exploration
            result1 = minimize(
                _soft_min_objective,
                init_uv.flatten(),
                args=(beta1, 100.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Check time before stage 2
            if time.time() > deadline - 5.0:
                return _clamp_bary(result1.x.reshape(N, 2))
            
            # Stage 2: Exploitation
            result2 = minimize(
                _soft_min_objective,
                result1.x,
                args=(beta2, 10000.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            return _clamp_bary(result2.x.reshape(N, 2))
            
        except Exception:
            return init_uv
    
    def _evaluate_and_update(bary):
        """Evaluate the hard min area and update best if improved."""
        nonlocal best_score, best_uv
        bary = _clamp_bary(bary)
        areas = _calc_areas_bary(bary)
        score = np.min(areas)
        
        if score > best_score:
            best_score = score
            best_uv = bary.copy()
            return True
        return False
    
    # 1. Start with hexagonal lattice initialization
    hex_init = _generate_hexagonal_init()
    optimized = _try_optimize(hex_init, beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    _evaluate_and_update(optimized)
    
    # 2. Try structured initializations
    for init_uv in _generate_structured_init():
        if time.time() > deadline - 40.0:
            break
        optimized = _try_optimize(init_uv, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        _evaluate_and_update(optimized)
    
    # 3. Try multiple random initializations
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 30.0:
            break
        rng_seed = 100 + i
        rand_init = _generate_random_init(seed=rng_seed)
        optimized = _try_optimize(rand_init, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        _evaluate_and_update(optimized)
    
    # 4. Perturb the best found and re-optimize
    if best_uv is not None:
        for i in range(5):
            if time.time() > deadline - 15.0:
                break
            perturbed = _generate_perturbed_init(best_uv, noise_scale=0.02, seed=200 + i)
            optimized = _try_optimize(perturbed, beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
            _evaluate_and_update(optimized)
    
    # 5. Final polish with Nelder-Mead on hard min
    if best_uv is not None and time.time() < deadline - 10.0:
        try:
            nm_result = minimize(
                _hard_min_objective,
                best_uv.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            _evaluate_and_update(nm_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 6. One more L-BFGS-B polish with very high beta
    if best_uv is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(
                _soft_min_objective,
                best_uv.flatten(),
                args=(5000.0, 1e6),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            _evaluate_and_update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # If no valid config was found (shouldn't happen), return fallback
    if best_uv is None:
        rng = np.random.default_rng(123)
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        best_uv = np.column_stack([u, v])
    
    # Convert best barycentric to Cartesian
    final_uv = _clamp_bary(best_uv)
    points = _bary_to_cart(final_uv)
    
    return points

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Points:\n{pts}")
    # Convert back to barycentric for area calculation
    # Actually, let's just compute areas directly from Cartesian
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    print(f"Min Area: {np.min(areas):.6f}")
