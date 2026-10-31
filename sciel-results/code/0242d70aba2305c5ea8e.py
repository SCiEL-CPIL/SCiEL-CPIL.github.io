import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _logsumexp_smooth_obj(x_flat, temperature):
    """
    Smooth surrogate: -log-sum-exp of triangle areas / hull area.
    As temperature -> 0, this approaches -min(area)/hull_area.
    Returns a smooth objective for gradient-based methods.
    """
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    # Normalize by hull area to get dimensionless areas
    norm_areas = areas / hull_area
    
    # Soft-min via log-sum-exp: -T * log(sum(exp(-a_i/T))) ≈ min(a_i) as T->0
    # We want to maximize min, so minimize -soft_min
    # soft_min = -T * log(sum(exp(-a_i/T)))
    # We want to minimize -soft_min = T * log(sum(exp(-a_i/T)))
    
    if temperature < 1e-8:
        return -np.min(norm_areas)
    
    # Numerically stable log-sum-exp
    scaled = -norm_areas / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * log_sum_exp
    return -soft_min

def _logsumexp_smooth_grad(x_flat, temperature):
    """
    Gradient of the smooth objective.
    """
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return np.zeros_like(x_flat)
    
    areas = _triangle_areas_vectorized(points)
    norm_areas = areas / hull_area
    
    if temperature < 1e-8:
        # Fallback to subgradient of min
        min_idx = np.argmin(norm_areas)
        grad = np.zeros_like(x_flat)
        i, j, k = _TRIPLE_IDX[min_idx]
        pi = points[i]
        pj = points[j]
        pk = points[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign = np.sign(cross) if cross != 0 else 1.0
        
        # d(area)/d(points) for triangle (i,j,k)
        # area = 0.5 * |cross(v1, v2)|
        # d(area)/dp_i = -0.5 * sign * [v2, -v1] (perpendicular)
        # d(area)/dp_j = 0.5 * sign * [v2, -v1]
        # d(area)/dp_k = 0.5 * sign * [-v2, v1]
        # Actually, let's compute properly:
        # area = 0.5 * |x1(y2-y3) - y1(x2-x3) + x2(y3-y1) - y2(x3-x1) + x3(y1-y2) - y3(x1-x2)|
        # This is complex; use numerical gradient for robustness
        eps = 1e-7
        grad = np.zeros_like(x_flat)
        for d in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_plus[d] += eps
            f_plus = _logsumexp_smooth_obj(x_plus, temperature)
            grad[d] = (f_plus - _logsumexp_smooth_obj(x_flat, temperature)) / eps
        return grad
    
    scaled = -norm_areas / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # d(soft_min)/d(area_m) = -weights[m]
    # d(area_m)/d(points) needs to be computed
    # For simplicity, use finite differences for the gradient
    eps = 1e-7
    grad = np.zeros_like(x_flat)
    base_val = _logsumexp_smooth_obj(x_flat, temperature)
    for d in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_plus[d] += eps
        grad[d] = (_logsumexp_smooth_obj(x_plus, temperature) - base_val) / eps
    return grad

def _make_ring_points(n_out, n_in, r_in, delta):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(delta, 2 * np.pi + delta, n_in, endpoint=False)
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    return np.vstack([pts_out, pts_in])

def _generate_diverse_seeds(deadline):
    """Generate diverse initial point configurations."""
    seeds = []
    t_start = time.time()
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    seeds.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two-ring configurations with various parameters
    for n_out, n_in in [(7, 6), (6, 7), (5, 8), (8, 5)]:
        for r_in in [0.3, 0.5, 0.7]:
            for delta in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                if time.time() > deadline - 30:
                    return seeds
                pts = _make_ring_points(n_out, n_in, r_in, delta)
                seeds.append(pts)
    
    # 3. Random configurations with scaling
    rng = np.random.RandomState(42)
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        pts = rng.randn(N, 2)
        # Normalize to unit area
        hull_area = _hull_area(pts)
        if hull_area > 1e-6:
            pts = pts / np.sqrt(hull_area)
            seeds.append(pts)
    
    # 4. Perturbed regular polygon
    for scale in [0.8, 1.0, 1.2]:
        if time.time() > deadline - 30:
            break
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = scale + rng.randn(N) * 0.1
        radii = np.clip(radii, 0.1, 2.0)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        hull_area = _hull_area(pts)
        if hull_area > 1e-6:
            pts = pts / np.sqrt(hull_area)
            seeds.append(pts)
    
    return seeds

def _anneal_optimize(points, deadline):
    """
    Optimize using annealing smooth surrogate + gradient-based L-BFGS-B.
    Start with high temperature (smooth), gradually decrease to sharp.
    """
    t_start = time.time()
    x = points.flatten().copy()
    best_score = _score(points)
    best_x = x.copy()
    
    # Annealing schedule: start smooth, end sharp
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() > deadline - 5:
            break
        
        # Use L-BFGS-B with smooth objective
        try:
            res = minimize(
                _logsumexp_smooth_obj,
                x,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x = res.x
            s = _score(x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = x.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead on the smooth objective for robustness
        if time.time() < deadline - 3:
            try:
                res_nm = minimize(
                    _logsumexp_smooth_obj,
                    x,
                    args=(temp,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                x = res_nm.x
                s = _score(x.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = x.copy()
            except Exception:
                pass
    
    return best_x.reshape(N, 2), best_score

def _final_polish(points, deadline):
    """Final high-precision polish using Nelder-Mead on the hard objective."""
    if time.time() > deadline - 3:
        return points, _score(points)
    
    x = points.flatten().copy()
    best_score = _score(points)
    best_x = x.copy()
    
    try:
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)) if _score(x.reshape(N, 2)) > 0 else 1e10,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        s = _score(res.x.reshape(N, 2))
        if s > best_score:
            best_score = s
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_points = None
    best_score = 0.0

    def _update_best(points, score=None):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        if score is None:
            score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate diverse seeds
    seeds = _generate_diverse_seeds(deadline)
    
    # Anneal-optimize each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 15:
            break
        optimized, score = _anneal_optimize(seed, deadline)
        _update_best(optimized, score)
        print(f"Seed {i}: score={score:.6f}", flush=True)

    # Final polish on the best
    if best_points is not None and time.time() < deadline - 3:
        polished, polished_score = _final_polish(best_points, deadline)
        _update_best(polished, polished_score)
        print(f"Final polished: score={polished_score:.6f}", flush=True)

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
