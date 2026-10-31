import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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

def _smooth_objective(params, p=40.0):
    """Smooth surrogate: minimize -soft_min(areas)/hull_area using power mean of inverse areas."""
    points = params.reshape(N, 2)
    
    # Center points to avoid numerical issues
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    hull_area = _hull_area(centered)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(centered)
    areas = np.maximum(areas, 1e-15)
    
    inv_areas = 1.0 / areas
    inv_areas_p = inv_areas ** p
    mean_inv_p = np.mean(inv_areas_p)
    soft_min = mean_inv_p ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _exact_objective(params):
    """Exact objective: minimize -score."""
    points = params.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _make_ring_config(n_outer, r_inner, offset_angle=0.0):
    """Create a two-ring configuration: n_outer points on unit circle, rest on inner circle."""
    n_inner = N - n_outer
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_outer + offset_angle
    
    radii_outer = np.ones(n_outer)
    radii_inner = np.full(n_inner, r_inner)
    
    angles = np.concatenate([angles_outer, angles_inner])
    radii = np.concatenate([radii_outer, radii_inner])
    
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    return np.column_stack([xs, ys])

def _make_regular_polygon():
    """Regular 13-gon on unit circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _make_grid_base():
    """4x4 grid minus one corner, centered."""
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[:-1]
    pts = pts - np.mean(pts, axis=0)
    return pts

def _coordinate_descent_polish(points, deadline, max_rounds=50):
    """Coordinate descent: optimize each point individually while fixing others."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            p_i = best_points[i].copy()
            
            def obj_1d(x):
                candidate = best_points.copy()
                candidate[i] = x
                s = _score(candidate)
                if s <= 0:
                    return 1e10
                return -s
            
            try:
                res = minimize(
                    obj_1d,
                    p_i,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                new_score = -res.fun
                if new_score > best_score + 1e-12:
                    best_points[i] = res.x
                    best_score = new_score
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return best_points, best_score

def _simulated_annealing(points, deadline, max_iter=20000):
    """Simulated annealing with adaptive step size."""
    best_points = points.copy()
    best_score = _score(best_points)
    current_points = points.copy()
    current_score = best_score
    
    t_start = time.time()
    T = 0.5
    T_min = 0.001
    alpha = 0.999
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        scale = T * 0.1
        perturbation = rng.normal(0, scale, size=(N, 2))
        candidate = current_points + perturbation
        
        candidate_score = _score(candidate)
        
        if candidate_score <= 0:
            delta = 1e10
        else:
            delta = current_score - candidate_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_points = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_points = candidate.copy()
        
        T *= alpha
    
    return best_points, best_score

def _local_polish(points, deadline):
    """Multi-stage local polish: smooth gradient, then exact Nelder-Mead, then coordinate descent."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Stage 1: L-BFGS-B on smooth surrogate
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=40.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead on exact objective
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Stage 3: Coordinate descent
    if time.time() < deadline - 5:
        try:
            cd_points, cd_score = _coordinate_descent_polish(best_points, deadline, max_rounds=30)
            if cd_score > best_score:
                best_score = cd_score
                best_points = cd_points.copy()
        except Exception:
            pass
    
    # Stage 4: Simulated annealing
    if time.time() < deadline - 3:
        try:
            sa_points, sa_score = _simulated_annealing(best_points, deadline, max_iter=5000)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points.copy()
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Phase 1: Differential Evolution in Cartesian space
    if time.time() < deadline - 40:
        bounds = [(-1.5, 1.5)] * (2 * N)
        
        max_iter = max(20, min(100, int((deadline - time.time()) / 5.0)))
        
        de_params = {
            'maxiter': max_iter,
            'popsize': 18,
            'mutation': (0.5, 0.9),
            'recombination': 0.7,
            'tol': 1e-6,
            'seed': 42,
            'polish': False,
            'init': 'sobol',
            'updating': 'immediate',
        }
        
        try:
            result = differential_evolution(
                _exact_objective,
                bounds,
                **de_params
            )
            if result.fun < 1e9:
                de_points = result.x.reshape(N, 2)
                _update_best(de_points)
                print(f"DE: score={best_score:.6f}", flush=True)
        except Exception:
            pass

    # Phase 2: Local polish from DE result
    if best_points is not None and time.time() < deadline - 30:
        polished_points, polished_score = _local_polish(best_points, deadline)
        _update_best(polished_points)
        print(f"Polished DE: score={best_score:.6f}", flush=True)

    # Phase 3: Diverse structured starts with local polish
    rng = np.random.default_rng(12345)
    
    structured_inits = []
    
    # Regular 13-gon
    structured_inits.append(_make_regular_polygon())
    
    # Two-ring configurations
    for n_outer in [5, 6, 7, 8]:
        for r_inner in [0.3, 0.5, 0.6, 0.7, 0.8]:
            for offset in [0.0, np.pi / n_outer / 2]:
                structured_inits.append(_make_ring_config(n_outer, r_inner, offset))
    
    # Grid base
    structured_inits.append(_make_grid_base())
    
    # Random starts
    for i in range(8):
        angles = np.sort(rng.random(N) * 2 * np.pi)
        radii = 0.5 + rng.random(N) * 1.0
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        structured_inits.append(np.column_stack([xs, ys]))
    
    for i, init_points in enumerate(structured_inits):
        if time.time() > deadline - 25:
            break
        
        polished_points, polished_score = _local_polish(init_points, deadline)
        _update_best(polished_points)
        
        if i % 4 == 0:
            print(f"Start {i}: score={best_score:.6f}", flush=True)

    # Phase 4: Additional simulated annealing from best
    if best_points is not None and time.time() < deadline - 10:
        sa_points, sa_score = _simulated_annealing(best_points, deadline, max_iter=15000)
        _update_best(sa_points)
        print(f"After SA: score={best_score:.6f}", flush=True)

    # Phase 5: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        final_points, final_score = _local_polish(best_points, deadline)
        _update_best(final_points)
        print(f"Final: score={best_score:.6f}", flush=True)

    if best_points is None:
        best_points = _make_regular_polygon()

    return best_points
