"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _soft_min_area(points, beta=100.0):
    """Compute soft-min of triangle areas using log-sum-exp for smoothness."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft min via log-sum-exp: -log(sum(exp(-beta*a)))/beta
    # Use max for numerical stability
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - log_sum / beta
    return soft_min


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _params_to_points(params):
    """Convert 2*N params to N points inside triangle using barycentric coords."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        points[i] = np.array([u * 1.0 + v * 0.5, v * H])
    return points


def _objective(params, beta=100.0):
    """Minimize negative soft-min-area (smooth surrogate)."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_soft_min_area(points, beta)


def _objective_exact(params):
    """Minimize negative exact min-area."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    bounds = [(0.0, 1.0)] * (2 * N)

    # Phase 1: Differential Evolution with smooth objective
    try:
        def _de_objective(params):
            return _objective(params, beta=100.0)
        
        result = differential_evolution(
            _de_objective,
            bounds=bounds,
            maxiter=150,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        if time.time() > deadline:
            if best_params is not None:
                return _params_to_points(best_params)
            rng = np.random.RandomState(42)
            pts = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        pts.append([u + 0.5 * v, H * v])
                        break
            return np.array(pts)
        best_params = result.x
        best_score = -_objective_exact(best_params)
    except Exception:
        pass

    # Phase 2: Polish with smooth objective, then exact
    if time.time() < deadline - 30.0:
        try:
            res2 = minimize(
                lambda p: _objective(p, beta=100.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score2 = -_objective_exact(res2.x)
            if score2 > best_score:
                best_params = res2.x
                best_score = score2
        except Exception:
            pass

    # Phase 3: Polish with exact objective
    if time.time() < deadline - 15.0:
        try:
            res3 = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score3 = -res3.fun
            if score3 > best_score:
                best_params = res3.x
                best_score = score3
        except Exception:
            pass

    # Phase 4: Random restarts with smooth then exact polish
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(123)
        for restart in range(8):
            if time.time() > deadline - 10.0:
                break
            params = np.zeros(2 * N)
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        params[2 * i] = u
                        params[2 * i + 1] = v
                        break
            try:
                # Smooth polish first
                res_s = minimize(
                    lambda p: _objective(p, beta=100.0),
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                # Then exact polish
                res_e = minimize(
                    _objective_exact,
                    res_s.x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                score_r = -res_e.fun
                if score_r > best_score:
                    best_params = res_e.x
                    best_score = score_r
            except Exception:
                pass

    if best_params is not None:
        points = _params_to_points(best_params)
        for i in range(N):
            if not _is_inside(points[i]):
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    pts.append([u + 0.5 * v, H * v])
                    break
        return np.array(pts)

# EVOLVE-BLOCK-END
