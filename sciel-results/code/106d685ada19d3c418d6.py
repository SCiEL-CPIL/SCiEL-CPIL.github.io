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


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
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


# ---------- Polar parameterization (from inspiration 2) ----------

def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points.
    polar shape: (2N,) -> angles[0:N], radii[N:2N]
    """
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _cartesian_to_polar(points):
    """Convert Cartesian points to polar (angles, radii)."""
    # Center first
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    radii = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
    # Sort by angle for consistent parameterization
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = radii[order]
    # Make angles continuous
    for i in range(1, len(angles_sorted)):
        if angles_sorted[i] - angles_sorted[i-1] < 0:
            angles_sorted[i:] += 2 * np.pi
    return np.concatenate([angles_sorted, radii_sorted])


def _polar_objective(polar, temperature=None):
    """Objective for polar-parameterized optimization."""
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    if temperature is not None:
        return _neg_soft_score(points, temperature)
    return _neg_score(points)


# ---------- Seed generators ----------

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


def _seed_quasi_regular(rng_seed=42, perturbation=0.03):
    """Quasi-regular: regular polygon with small random perturbations."""
    rng = np.random.RandomState(rng_seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + perturbation * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize_cartesian(points, deadline, max_temp=0.05, min_temp=1e-6, n_steps=12):
    """
    Anneal the soft-min temperature and optimize at each step in Cartesian space.
    Uses L-BFGS-B then Powell at each temperature level.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on soft objective
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on soft objective
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj2, best.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-11},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


def _annealed_soft_minimize_polar(points, deadline, max_temp=0.05, min_temp=1e-6, n_steps=10):
    """
    Anneal the soft-min temperature and optimize in polar space.
    This enforces a sorted angular structure which may help avoid degenerate configurations.
    """
    best = points.copy()
    best_s = _score(best)
    best_polar = _cartesian_to_polar(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # Powell in polar space
        if time.time() < deadline - 5:
            try:
                def obj_polar(polar):
                    return _polar_objective(polar, temperature=temp)
                
                res = minimize(
                    obj_polar, best_polar,
                    method='Powell',
                    options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand_polar = res.x
                cand = _polar_to_cartesian(cand_polar)
                cand = cand - np.mean(cand, axis=0)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                    best_polar = cand_polar.copy()
            except Exception:
                pass
        
        # Nelder-Mead in polar space
        if time.time() < deadline - 4:
            try:
                def obj_polar2(polar):
                    return _polar_objective(polar, temperature=temp)
                
                res = minimize(
                    obj_polar2, best_polar,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
                )
                cand_polar = res.x
                cand = _polar_to_cartesian(cand_polar)
                cand = cand - np.mean(cand, axis=0)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                    best_polar = cand_polar.copy()
            except Exception:
                pass
    
    return best, best_s


def _polish_true_objective(points, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    # Powell
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _polish_polar_true(points, deadline):
    """Polish in polar space using the true objective."""
    best = points.copy()
    best_s = _score(best)
    best_polar = _cartesian_to_polar(best)
    
    # Powell in polar
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda polar: _polar_objective(polar),
                best_polar,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11},
            )
            cand_polar = res.x
            cand = _polar_to_cartesian(cand_polar)
            cand = cand - np.mean(cand, axis=0)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
                best_polar = cand_polar.copy()
        except Exception:
            pass
    
    # Nelder-Mead in polar
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda polar: _polar_objective(polar),
                best_polar,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            cand_polar = res.x
            cand = _polar_to_cartesian(cand_polar)
            cand = cand - np.mean(cand, axis=0)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _local_search(points, deadline, n_iter=300, sigma_start=0.02, sigma_end=0.0005):
    """
    Coordinate-wise local search with adaptive step size.
    Tries moving each point in random directions, keeping improvements.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        
        # Adaptive sigma
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        # Pick a random point and a random direction
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        # Try a few step sizes
        for scale in [1.0, 0.5, 0.2, 2.0, 0.1]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def _coordinate_descent_polish(points, deadline, n_sweeps=20):
    """
    Coordinate descent: optimize each point's (x,y) one at a time using Nelder-Mead.
    This is a strong local refinement that can escape shallow local minima.
    """
    best = points.copy()
    best_s = _score(best)
    
    for sweep in range(n_sweeps):
        if time.time() >= deadline - 5:
            break
        
        for i in range(N):
            if time.time() >= deadline - 5:
                break
            
            # Get current position of point i
            pt_i = best[i].copy()
            
            # Define sub-objective: only point i moves
            def sub_obj(x_i):
                cand = best.copy()
                cand[i] = x_i
                s = _score(cand)
                if s <= 0:
                    return 1e10
                return -s
            
            try:
                res = minimize(
                    sub_obj, pt_i,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
                )
                new_pt = res.x
                cand = best.copy()
                cand[i] = new_pt
                s = _score(cand)
                if s > best_s + 1e-15:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


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
        for r_in in [0.3, 0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out)]:
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
    
    # Quasi-regular with different seeds and perturbation levels
    for seed_val in [42, 123, 7, 99]:
        for pert in [0.02, 0.05, 0.1]:
            seeds.append(_seed_quasi_regular(seed_val, pert))
    
    # Evaluate all seeds
    for seed in seeds:
        _update(seed)
    
    if best_points is None:
        best_points = _seed_regular()
        best_score = _score(best_points)
    
    # ---------- Annealed soft-min optimization from top seeds ----------
    # Sort seeds by score and optimize from the top few
    seed_scores = [(seed, _score(seed)) for seed in seeds]
    seed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Optimize from top seeds in both Cartesian and Polar spaces
    n_optimize_cart = min(4, len(seed_scores))
    n_optimize_polar = min(3, len(seed_scores))
    
    # Cartesian optimization
    for i in range(n_optimize_cart):
        if time.time() >= deadline - 40:
            break
        seed_pts, seed_s = seed_scores[i]
        try:
            pts, s = _annealed_soft_minimize_cartesian(seed_pts, deadline, 
                                                       max_temp=0.05, min_temp=1e-6, n_steps=10)
            _update(pts)
        except Exception:
            pass
    
    # Polar optimization (different search space, may find different optima)
    for i in range(n_optimize_polar):
        if time.time() >= deadline - 25:
            break
        seed_pts, seed_s = seed_scores[i]
        try:
            pts, s = _annealed_soft_minimize_polar(seed_pts, deadline, 
                                                   max_temp=0.05, min_temp=1e-6, n_steps=8)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Polish the best found so far (Cartesian) ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _polish_true_objective(best_points, deadline)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Polish in Polar space ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _polish_polar_true(best_points, deadline)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Coordinate descent polish ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _coordinate_descent_polish(best_points, deadline, n_sweeps=15)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Local search from the best ----------
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _local_search(best_points, deadline, n_iter=200, 
                                    sigma_start=0.015, sigma_end=0.0005)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Final ultra-fine Nelder-Mead on best ----------
    if best_points is not None and time.time() < deadline - 3:
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
    
    # ---------- Final coordinate descent sweep ----------
    if best_points is not None and time.time() < deadline - 2:
        try:
            pts, s = _coordinate_descent_polish(best_points, deadline, n_sweeps=10)
            _update(pts)
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
