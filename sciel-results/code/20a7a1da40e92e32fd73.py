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
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _epigraph_objective(x, t):
    """
    Epigraph formulation: minimize t subject to all triangle areas >= t * hull_area.
    We reformulate as: minimize -t + penalty for violations.
    x = [p1, p2, ..., p13, t] where t is the auxiliary variable.
    """
    pts = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(pts)
    # We want all areas >= t * hull_area
    # Penalty for violations
    violations = np.maximum(0, t * hull_area - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + 1e6 * penalty


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    # Use stable log-sum-exp
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(jump=1.0):
    """Quasi-regular: regular polygon with small random perturbations."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = np.ones(N)
    # Small radial perturbations
    r += 0.02 * np.random.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _annealed_soft_minimize(x0, deadline, max_temp=0.1, min_temp=1e-4, n_steps=8):
    """
    Anneal the soft-min temperature and optimize at each step.
    Returns the best point configuration found.
    """
    best_points = x0.copy()
    best_score = _score(best_points)
    
    # Log-linear temperature schedule
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() > deadline - 10:
            break
        
        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, temp)
        
        # Try L-BFGS-B first (fast for smooth problems)
        try:
            res = minimize(
                obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
        
        # Then Powell for more robustness
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    obj,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass
    
    return best_points, best_score


def _perturb_restart(points, sigma=0.05):
    """Generate a perturbed version of the given points."""
    return points + sigma * np.random.randn(*points.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill
    
    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # ---------- Generate diverse seeds ----------
    seeds = []
    
    # Regular 13-gon
    seeds.append(_seed_regular())
    
    # Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2 * n_out)]:
                seeds.append(_seed_two_ring(n_out, r_in, a_off))
    
    # Three-ring configurations
    three_ring_configs = [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
        (6, 5, 2, 0.5, 0.25, np.pi / 6, 0.0),
    ]
    for (n1, n2, n3, r2, r3, o2, o3) in three_ring_configs:
        seeds.append(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))
    
    # Quasi-regular with random perturbations
    for _ in range(5):
        seeds.append(_seed_quasi_regular())
    
    # ---------- Evaluate all seeds ----------
    for seed in seeds:
        _update(seed)
    
    if best_points is None:
        best_points = _seed_regular()
        best_score = _score(best_points)
    
    # ---------- Annealed soft-min optimization from best seed ----------
    if time.time() < deadline - 20:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline, max_temp=0.15, min_temp=1e-5, n_steps=10)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Perturbation restarts from best solution ----------
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline - 12:
            break
        
        sigma = 0.02 + 0.012 * i  # Increasing perturbation size
        perturbed = _perturb_restart(best_points, sigma)
        
        # Quick annealed optimization from perturbed start
        remaining_budget = max(10, deadline - time.time() - 5)
        try:
            pts, s = _annealed_soft_minimize(perturbed, time.time() + remaining_budget * (1.0 - 0.08 * i), 
                                             max_temp=0.08, min_temp=1e-5, n_steps=7)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Final L-BFGS-B polish on the true (non-smooth) objective ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # ---------- Final Powell polish ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # ---------- Epigraph optimization ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            # Initialize t with current score
            t_init = best_score * 0.9
            x0_ep = np.concatenate([best_points.flatten(), [t_init]])
            
            res = minimize(
                _epigraph_objective,
                x0_ep,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
            )
            cand_pts = res.x[:2*N].reshape(N, 2)
            _update(cand_pts)
        except Exception:
            pass
    
    # ---------- Final ultra-fine Nelder-Mead ----------
    if best_points is not None and time.time() < deadline - 1.5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
