"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    Returns the soft-min ratio (higher is better).
    """
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

def _negative_soft_min_objective(x, temperature):
    """Objective for minimization: negative soft-min ratio."""
    pts = x.reshape(-1, 2)
    val = _soft_min_ratio(pts, temperature)
    if val < -1e8:
        return 1e8
    return -val

def _halton_sequence(n, base):
    """Generate first n Halton sequence values with given base."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        idx = i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result.append(r)
    return np.array(result)

def _halton_points_2d(n, seed=0):
    """Generate n 2D points using Halton sequences (bases 2 and 3)."""
    rng = np.random.RandomState(seed)
    offset = rng.randint(1, 1000)
    h2 = _halton_sequence(n + offset, 2)
    h3 = _halton_sequence(n + offset, 3)
    pts = np.column_stack([h2[offset:offset + n], h3[offset:offset + n]])
    return pts

def _concentric_ring_init(seed=0, n_inner=5, n_outer=8):
    """Generate points on two concentric rings for diverse initialization."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    # Inner ring
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _elliptic_regular_init(a, b, c, angle):
    """Apply elliptical distortion to a regular 13-gon.
    Parameters a, b, c define the symmetric matrix [[a, c/2], [c/2, b]].
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Rotate by angle
    theta = angle
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    base_pts = base_pts @ rot.T
    # Apply elliptical transform
    M = np.array([[a, c / 2], [c / 2, b]])
    pts = base_pts @ M.T
    return pts

def _elliptic_objective(params, temperature=0.01):
    """Objective for elliptic parameter optimization.
    params = [a, b, c, angle]
    """
    a, b, c, angle = params
    # Ensure positive definiteness
    if a <= 0 or b <= 0 or a * b - (c / 2) ** 2 <= 0:
        return 1e6
    try:
        pts = _elliptic_regular_init(a, b, c, angle)
    except Exception:
        return 1e6
    return _negative_soft_min_objective(pts.flatten(), temperature)

def _optimize_elliptic(start_params, temperature=0.01, max_iter=200):
    """Optimize elliptic parameters."""
    try:
        result = minimize(
            _elliptic_objective,
            start_params,
            args=(temperature,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.success:
            a, b, c, angle = result.x
            if a > 0 and b > 0 and a * b - (c / 2) ** 2 > 0:
                pts = _elliptic_regular_init(a, b, c, angle)
                return pts
    except Exception:
        pass
    return None

def _support_function_init(k, seed=0):
    """Generate points from a support function with Fourier basis.
    k: number of Fourier modes to use (plus DC term).
    """
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Support function: h(theta) = 1 + sum_{m=1}^{k} (a_m cos(m*theta) + b_m sin(m*theta))
    h = np.ones(N)
    for m in range(1, k + 1):
        a_m = rng.uniform(-0.3, 0.3)
        b_m = rng.uniform(-0.3, 0.3)
        h += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
    # Ensure h is positive
    h = np.maximum(h, 0.5)
    pts = np.column_stack([h * np.cos(angles), h * np.sin(angles)])
    return pts

def _support_function_objective(params, k=3, temperature=0.01):
    """Objective for support function parameter optimization.
    params: [a1, b1, a2, b2, ..., ak, bk]
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.ones(N)
    for m in range(1, k + 1):
        a_m = params[2*(m-1)]
        b_m = params[2*(m-1)+1]
        h += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
    h = np.maximum(h, 0.5)
    pts = np.column_stack([h * np.cos(angles), h * np.sin(angles)])
    return _negative_soft_min_objective(pts.flatten(), temperature)

def _optimize_support_function(k=3, seed=0, temperature=0.01, max_iter=200):
    """Optimize support function parameters."""
    rng = np.random.RandomState(seed)
    init_params = rng.uniform(-0.2, 0.2, 2*k)
    try:
        result = minimize(
            _support_function_objective,
            init_params,
            args=(k, temperature),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.success:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            h = np.ones(N)
            for m in range(1, k + 1):
                a_m = result.x[2*(m-1)]
                b_m = result.x[2*(m-1)+1]
                h += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
            h = np.maximum(h, 0.5)
            pts = np.column_stack([h * np.cos(angles), h * np.sin(angles)])
            return pts
    except Exception:
        pass
    return None

def _angular_to_cartesian(angles, radii):
    """Convert angles and radii to Cartesian coordinates."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _cartesian_to_angular(pts):
    """Convert Cartesian to angles and radii."""
    radii = np.linalg.norm(pts, axis=1)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    angles = angles % (2 * np.pi)
    return angles, radii

def _local_search_angular(angles, radii, time_budget, rng):
    """
    Coordinate-wise local search in angular parameterization.
    Perturb one angle or radius at a time.
    """
    start_time = time.time()
    best_angles = angles.copy()
    best_radii = radii.copy()
    best_score = _score(_angular_to_cartesian(best_angles, best_radii))
    
    # Adaptive step sizes
    angle_step = 0.02
    radius_step = 0.02
    
    while time.time() - start_time < time_budget - 1:
        improved = False
        for i in range(N):
            # Try perturbing angle
            for delta in [-angle_step, angle_step]:
                trial_angles = best_angles.copy()
                trial_angles[i] += delta
                trial_scores = _score(_angular_to_cartesian(trial_angles, best_radii))
                if trial_scores > best_score:
                    best_angles = trial_angles
                    best_score = trial_scores
                    improved = True
            
            # Try perturbing radius
            for delta in [-radius_step, radius_step]:
                trial_radii = best_radii.copy()
                trial_radii[i] = np.maximum(0.01, trial_radii[i] + delta)
                trial_scores = _score(_angular_to_cartesian(best_angles, trial_radii))
                if trial_scores > best_score:
                    best_radii = trial_radii
                    best_score = trial_scores
                    improved = True
        
        if not improved:
            # Reduce step sizes
            angle_step *= 0.5
            radius_step *= 0.5
            if angle_step < 1e-5:
                break
    
    return best_angles, best_radii, best_score

def _refine_with_annealing(pts, time_budget):
    """
    Refine a point set using multi-stage temperature annealing.
    Start with higher temperature for exploration, lower for exploitation.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    # Stage 1: Higher temperature for broader exploration
    if time.time() - t0 < time_budget * 0.4:
        temp1 = 0.05
        def obj1(x):
            p = x.reshape(N, 2)
            val = _soft_min_ratio(p, temp1)
            if val < -1e8:
                return 1e8
            return -val
        try:
            res = minimize(obj1, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Stage 2: Medium temperature
    if time.time() - t0 < time_budget * 0.7:
        temp2 = 0.01
        def obj2(x):
            p = x.reshape(N, 2)
            val = _soft_min_ratio(p, temp2)
            if val < -1e8:
                return 1e8
            return -val
        try:
            res = minimize(obj2, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-9})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Stage 3: Low temperature for fine-tuning
    if time.time() - t0 < time_budget - 1:
        temp3 = 0.002
        def obj3(x):
            p = x.reshape(N, 2)
            val = _soft_min_ratio(p, temp3)
            if val < -1e8:
                return 1e8
            return -val
        try:
            res = minimize(obj3, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with annealed L-BFGS on full 2D coordinates
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
        lambda s: _support_function_init(3, seed=s),
        lambda s: _support_function_init(2, seed=s + 100),
    ]
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            # Refine with annealing
            refined_pts, refined_score = _refine_with_annealing(pts0, time_budget=30)
            
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
        except Exception:
            continue
    
    # Phase 2: Elliptic parameter optimization for structured solutions
    if time.time() < deadline - 25:
        # Try several elliptic parameter starts
        elliptic_starts = [
            [1.0, 1.0, 0.0, 0.0],
            [1.2, 0.8, 0.0, 0.0],
            [0.8, 1.2, 0.0, 0.0],
            [1.0, 1.0, 0.3, 0.0],
            [1.1, 0.9, -0.2, np.pi/6],
            [0.9, 1.1, 0.2, -np.pi/6],
        ]
        for sp in elliptic_starts:
            if time.time() > deadline - 15:
                break
            try:
                pts = _optimize_elliptic(sp, temperature=0.01, max_iter=150)
                if pts is not None:
                    # Further refine
                    refined_pts, refined_score = _refine_with_annealing(pts, time_budget=15)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_pts = refined_pts.copy()
            except Exception:
                continue
    
    # Phase 2b: Support function optimization
    if time.time() < deadline - 20:
        for k in [2, 3]:
            for seed in range(3):
                if time.time() > deadline - 10:
                    break
                try:
                    pts = _optimize_support_function(k=k, seed=seed, temperature=0.01, max_iter=150)
                    if pts is not None:
                        refined_pts, refined_score = _refine_with_annealing(pts, time_budget=10)
                        if refined_score > best_score:
                            best_score = refined_score
                            best_pts = refined_pts.copy()
                except Exception:
                    continue
    
    # Phase 3: Angular local search on the best solution
    if time.time() < deadline - 15:
        try:
            angles, radii = _cartesian_to_angular(best_pts)
            rng = np.random.RandomState(42)
            best_angles, best_radii, ang_score = _local_search_angular(
                angles, radii, time_budget=10, rng=rng
            )
            ang_pts = _angular_to_cartesian(best_angles, best_radii)
            if ang_score > best_score:
                best_score = ang_score
                best_pts = ang_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            # Use very low temperature for final polish
            temp_final = 0.0005
            def obj_final(x):
                p = x.reshape(N, 2)
                val = _soft_min_ratio(p, temp_final)
                if val < -1e8:
                    return 1e8
                return -val
            
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
            final_pts = res.x.reshape(N, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
