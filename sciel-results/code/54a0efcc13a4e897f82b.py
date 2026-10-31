import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


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
        if np.ptp(points, axis=0).max() < 1e-12:
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


def _neg_score(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _radial_to_cartesian(radii):
    """
    Convert radii to Cartesian points with fixed regular polygon angles.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _neg_score_radial(radii):
    points = _radial_to_cartesian(radii)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _optimize_radial(deadline, n_starts=10):
    """
    Multi-start Nelder-Mead optimization in the radial space (13 dimensions).
    """
    best_points = None
    best_score = -np.inf
    
    # Angles for conversion
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    rng = np.random.default_rng(42)
    
    # Define starting points for radial optimization
    starts = []
    # 1. Regular polygon (all radii 1)
    starts.append(np.ones(N))
    # 2. Alternating radii
    alt = np.ones(N)
    alt[::2] = 0.8
    alt[1::2] = 1.2
    starts.append(alt)
    # 3. Random perturbations of regular
    for _ in range(n_starts - 2):
        r = np.ones(N) + rng.normal(0, 0.2, N)
        r = np.clip(r, 0.1, 2.0)
        starts.append(r)
        
    for i, r0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        try:
            # Nelder-Mead on radial space
            res = minimize(
                _neg_score_radial,
                r0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            r_opt = res.x
            # Ensure positive radii
            r_opt = np.maximum(r_opt, 0.01)
            
            points = _radial_to_cartesian(r_opt)
            s = _score(points)
            
            if s > best_score:
                best_score = s
                best_points = points.copy()
                
        except Exception:
            pass
            
    return best_points, best_score


def _polish_cartesian(points, deadline):
    """
    Polish the points in Cartesian space using Nelder-Mead.
    """
    if points is None:
        return None, -np.inf
        
    x0 = points.flatten()
    best_s = _score(points)
    
    try:
        res = minimize(
            _neg_score,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            return cand, s
    except Exception:
        pass
        
    return points, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # 1. Radial Optimization (Coarse Search)
    # This restricts angles to regular polygon, which is a strong prior for symmetric optima.
    if time.time() < deadline - 30:
        try:
            rad_points, rad_score = _optimize_radial(deadline, n_starts=15)
            if rad_points is not None:
                _update_best(rad_points)
        except Exception:
            pass

    # 2. Cartesian Polish (Fine Search)
    # Allow angles to shift from the regular configuration.
    if best_points is not None and time.time() < deadline - 10:
        try:
            pol_points, pol_score = _polish_cartesian(best_points, deadline)
            _update_best(pol_points)
        except Exception:
            pass
            
    # 3. Local Search / Jittering
    # If we have time, do a simple stochastic local search on the best Cartesian point
    if best_points is not None and time.time() < deadline - 5:
        rng = np.random.default_rng(123)
        current = best_points.copy()
        current_s = best_score
        sigma = 0.05
        
        for i in range(200):
            if time.time() > deadline - 2:
                break
            
            # Random direction
            idx = rng.randint(N)
            direction = rng.randn(2)
            direction /= (np.linalg.norm(direction) + 1e-15)
            
            # Try a few step sizes
            for scale in [1.0, 0.5, 0.1, 2.0]:
                step = sigma * scale
                cand = current.copy()
                cand[idx] += step * direction
                
                s = _score(cand)
                if s > current_s:
                    current = cand
                    current_s = s
                    break
        
        _update_best(current)
        
        # Final polish after local search
        if time.time() < deadline - 2:
            try:
                pol_points, pol_score = _polish_cartesian(current, deadline)
                _update_best(pol_points)
            except Exception:
                pass

    if best_points is None:
        # Fallback
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_points
