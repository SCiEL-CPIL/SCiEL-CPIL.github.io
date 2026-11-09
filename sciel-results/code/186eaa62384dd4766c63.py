"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

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

# Precompute all C(11,2) pair index pairs
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)
_n_pairs = len(_pair_i)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _min_area_and_idx(points):
    """Compute minimum triangle area and which indices achieve it."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    min_idx = np.argmin(areas)
    return areas.min(), min_idx


def _soft_min_area(points, beta=50.0):
    """Compute soft-min of triangle areas using log-sum-exp for smoothness."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    soft_min = -max_a - (1.0 / beta) * np.log(np.sum(exp_terms))
    return soft_min


def _min_pairwise_dist(points):
    """Compute minimum pairwise distance among all C(N,2) pairs."""
    p1 = points[_pair_i]
    p2 = points[_pair_j]
    d = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    return d.min()


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


def _clamp_to_triangle(points):
    """Clamp points to be inside the equilateral triangle."""
    pts = points.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Clamp y >= 0
        if y < 0:
            y = 0
        # Clamp x >= 0, x <= 1
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        # Clamp y <= sqrt(3)*x (left edge)
        max_y_left = np.sqrt(3) * x
        if y > max_y_left:
            y = max_y_left
        # Clamp y <= sqrt(3)*(1-x) (right edge)
        max_y_right = np.sqrt(3) * (1 - x)
        if y > max_y_right:
            y = max_y_right
        pts[i] = [x, y]
    return pts


def _symmetric_params_to_points(params):
    """Map 11 parameters to 11 points with reflection symmetry about x=0.5.
    params[0]: y of center point (x=0.5)
    params[1:]: for 5 pairs, (x_i, y_i) where x_i in [0, 0.5], y_i in [0, H]
    Point i left: (x_i, y_i), right: (1-x_i, y_i)
    """
    points = np.zeros((N, 2))
    # Center point
    points[0, 0] = 0.5
    points[0, 1] = np.clip(params[0], 0.0, H)
    
    # 5 pairs
    for i in range(5):
        x = np.clip(params[1 + 2*i], 0.0, 0.5)
        y = np.clip(params[2 + 2*i], 0.0, H)
        # Clamp y to triangle boundary for this x
        max_y = np.sqrt(3) * min(x, 1.0 - x)
        if y > max_y:
            y = max_y
        points[1 + 2*i, 0] = x
        points[1 + 2*i, 1] = y
        points[2 + 2*i, 0] = 1.0 - x
        points[2 + 2*i, 1] = y
    return points


def _objective_symmetric(params):
    """Minimize negative exact min-area for symmetric config."""
    points = _symmetric_params_to_points(params)
    return -_min_area(points)


def _objective_soft_symmetric(params, beta=50.0):
    """Minimize negative soft-min area for symmetric config."""
    points = _symmetric_params_to_points(params)
    return -_soft_min_area(points, beta)


def _symmetric_bounds():
    """Bounds for symmetric parameters: y_center, then (x,y) for 5 pairs."""
    bounds = [(0.0, H)]
    for _ in range(5):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, H))
    return bounds


def _known_good_params():
    """Return parameters for a known good symmetric configuration."""
    # Based on known near-optimal arrangements for 11 points
    # Using a pattern that distributes points well
    params = np.zeros(11)
    
    # Center point at centroid
    params[0] = H / 3.0
    
    # 5 pairs - arranged to maximize minimum triangle area
    # Pair 1: near top
    params[1] = 0.28
    params[2] = H * 0.72
    
    # Pair 2: upper middle
    params[3] = 0.12
    params[4] = H * 0.55
    params[5] = 0.38
    params[6] = H * 0.55
    
    # Pair 3: middle
    params[7] = 0.08
    params[8] = H * 0.35
    params[9] = 0.42
    params[10] = H * 0.35
    
    return params


def _coarse_to_fine_symmetric(params, bounds, deadline, t_start):
    """Coarse-to-fine optimization: start with coarse NM, then refine."""
    best_params = params.copy()
    best_score = -_objective_symmetric(best_params)
    
    # Coarse phase
    if time.time() < deadline - 30:
        res = minimize(
            _objective_symmetric,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        score = -res.fun
        if score > best_score:
            best_params = res.x.copy()
            best_score = score
    
    # Fine phase
    if time.time() < deadline - 15:
        res = minimize(
            _objective_symmetric,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        score = -res.fun
        if score > best_score:
            best_params = res.x.copy()
            best_score = score
    
    # Ultra-fine phase
    if time.time() < deadline - 5:
        res = minimize(
            _objective_symmetric,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        score = -res.fun
        if score > best_score:
            best_params = res.x.copy()
            best_score = score
    
    return best_params, best_score


def _direct_points_to_params(points):
    """Convert direct points to symmetric params (for refinement)."""
    # Extract symmetric structure from points
    params = np.zeros(11)
    # Find center point (closest to x=0.5)
    dists_to_center = np.abs(points[:, 0] - 0.5)
    center_idx = np.argmin(dists_to_center)
    params[0] = points[center_idx, 1]
    
    # For remaining points, pair them by symmetry
    remaining = [i for i in range(N) if i != center_idx]
    # Sort by x coordinate
    remaining.sort(key=lambda i: points[i, 0])
    
    # Pair left and right
    for i in range(5):
        left_idx = remaining[i]
        right_idx = remaining[10 - i]
        x = min(points[left_idx, 0], points[right_idx, 0])
        y = (points[left_idx, 1] + points[right_idx, 1]) / 2.0
        params[1 + 2*i] = np.clip(x, 0.0, 0.5)
        params[2 + 2*i] = np.clip(y, 0.0, H)
    
    return params


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin for 150s limit

    best_params = None
    best_score = -np.inf
    bounds = _symmetric_bounds()
    
    def _update_best(params, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_params = params.copy()
            best_score = score

    # Phase 0: Start with known good configuration
    try:
        init_params = _known_good_params()
        score_init = -_objective_symmetric(init_params)
        _update_best(init_params, score_init)
    except Exception:
        pass

    # Phase 1: Differential Evolution with soft-min objective (global search)
    try:
        result_de = differential_evolution(
            lambda p: _objective_soft_symmetric(p, beta=50.0),
            bounds=bounds,
            maxiter=150,
            popsize=25,
            tol=1e-8,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        score_de = -_objective_symmetric(result_de.x)
        _update_best(result_de.x, score_de)
    except Exception:
        pass

    # Phase 2: Dual annealing for additional global exploration
    if time.time() < deadline - 50.0:
        try:
            result_da = dual_annealing(
                _objective_symmetric,
                bounds=bounds,
                maxiter=300,
                seed=123,
                restarts=3,
                polish=True
            )
            score_da = -result_da.fun
            _update_best(result_da.x, score_da)
        except Exception:
            pass

    # Phase 3: Coarse-to-fine from best solution
    if time.time() < deadline - 40.0 and best_params is not None:
        try:
            refined_params, refined_score = _coarse_to_fine_symmetric(
                best_params, bounds, deadline, t_start
            )
            _update_best(refined_params, refined_score)
        except Exception:
            pass

    # Phase 4: Multiple restarts with diverse initializations
    if time.time() < deadline - 25.0:
        rng = np.random.RandomState(777)
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline - 25.0:
                break
            try:
                if restart < 5 and best_params is not None:
                    # Perturb best solution with varying noise
                    noise_scale = 0.03 + 0.02 * restart
                    noise = rng.randn(11) * noise_scale
                    start = best_params + noise
                    for i, b in enumerate(bounds):
                        start[i] = np.clip(start[i], b[0], b[1])
                elif restart < 10:
                    # Perturb known good
                    start = _known_good_params() + rng.randn(11) * 0.1
                    for i, b in enumerate(bounds):
                        start[i] = np.clip(start[i], b[0], b[1])
                elif restart < 15:
                    # Random symmetric start
                    start = np.array([rng.uniform(b[0], b[1]) for b in bounds])
                else:
                    # Lattice-like start with randomization
                    start = _known_good_params()
                    # Randomly swap some pair positions
                    if rng.random() < 0.5:
                        i = rng.randint(0, 5)
                        j = rng.randint(0, 5)
                        start[1+2*i], start[1+2*j] = start[1+2*j], start[1+2*i]
                        start[2+2*i], start[2+2*j] = start[2+2*j], start[2+2*i]
                    start += rng.randn(11) * 0.05
                    for i, b in enumerate(bounds):
                        start[i] = np.clip(start[i], b[0], b[1])
                
                # Coarse-to-fine optimization
                res_params, res_score = _coarse_to_fine_symmetric(
                    start, bounds, deadline, t_start
                )
                _update_best(res_params, res_score)
            except Exception:
                pass

    # Phase 5: Final ultra-high-precision polish
    if time.time() < deadline - 3.0 and best_params is not None:
        try:
            res_final = minimize(
                _objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_f = -res_final.fun
            _update_best(res_final.x, score_f)
        except Exception:
            pass

    # Convert best params to points
    if best_params is not None:
        points = _symmetric_params_to_points(best_params)
        # Clamp to ensure inside triangle
        points = _clamp_to_triangle(points)
        return points
    else:
        # Fallback: use known good
        points = _symmetric_params_to_points(_known_good_params())
        points = _clamp_to_triangle(points)
        return points

# EVOLVE-BLOCK-END
