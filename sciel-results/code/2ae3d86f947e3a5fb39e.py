"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Combined strategy: radial-parameter convexity-preserving optimization (13 dims)
with log-sum-exp soft-min surrogate + L-BFGS-B, followed by full-2D polish and
targeted worst-triangle refinement."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TIME_LIMIT = 135.0  # safety margin before 150s kill

# Precompute all triangle index triples
_TRIS = np.array(list(combinations(range(N), 3)))
_N_TRIS = _TRIS.shape[0]

# EVOLVE-BLOCK-START

def _triangle_areas(pts):
    """Vectorized computation of all C(N,3) triangle areas."""
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min_score(pts, beta=50.0):
    """Soft-min surrogate: smooth approximation of min(areas)/hull_area."""
    try:
        areas = _triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        # Soft min using log-sum-exp trick
        min_a = np.min(areas)
        shifted = areas - min_a
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min_val = min_a - log_sum / beta
        return soft_min_val / hull_area
    except Exception:
        return 0.0

def _neg_soft_min_score(x, beta=50.0):
    """Negative soft-min score for minimization (full 2D)."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _neg_score(x):
    """Negative exact score for minimization (full 2D)."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _radial_to_pts(radii, angles):
    """Convert radial parameters (radii, angles) to 2D points."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _soft_min_radial(radii, beta=50.0):
    """Soft-min score computed from radial parameters (radii only, angles fixed)."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = _radial_to_pts(radii, angles)
    return _soft_min_score(pts, beta)

def _neg_soft_min_radial(x, beta=50.0):
    """Negative soft-min score for radial optimization."""
    radii = np.abs(x)  # enforce positive radii
    return -_soft_min_radial(radii, beta)

def _neg_score_radial(x):
    """Negative exact score for radial optimization."""
    radii = np.abs(x)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = _radial_to_pts(radii, angles)
    return -_score(pts)

def _generate_initializations():
    """Generate multiple diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_circle)
    
    # 2. Halton sequence in unit square
    def _halton(n, base):
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_temp = i
            while i_temp > 0:
                f /= base
                x += f * (i_temp % base)
                i_temp //= base
            result.append(x)
        return np.array(result)
    
    x = _halton(N, 2)
    y = _halton(N, 3)
    pts_halton = np.column_stack([x, y])
    inits.append(pts_halton)
    
    # 3. Concentric ring: 7 outer + 6 inner
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    inner_pts = 0.55 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    pts_rings = np.vstack([outer_pts, inner_pts])
    inits.append(pts_rings)
    
    # 4. Three concentric rings: 5 + 4 + 4
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 5
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    r1_pts = np.column_stack([np.cos(a1), np.sin(a1)])
    r2_pts = 0.65 * np.column_stack([np.cos(a2), np.sin(a2)])
    r3_pts = 0.35 * np.column_stack([np.cos(a3), np.sin(a3)])
    pts_3rings = np.vstack([r1_pts, r2_pts, r3_pts])
    inits.append(pts_3rings)
    
    # 5. Two rings with offset: 8 outer + 5 inner
    a_out = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    a_in = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 8
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = 0.5 * np.column_stack([np.cos(a_in), np.sin(a_in)])
    pts_2rings = np.vstack([outer, inner])
    inits.append(pts_2rings)
    
    # 6. Perturbed regular polygons
    rng = np.random.RandomState(42)
    for noise in [0.05, 0.1, 0.2, 0.3]:
        angles_p = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts_p = np.column_stack([np.cos(angles_p), np.sin(angles_p)])
        pts_p += rng.normal(0, noise, (N, 2))
        inits.append(pts_p)
    
    # 7. Random on unit circle
    for _ in range(5):
        angles_r = np.sort(rng.uniform(0, 2 * np.pi, N))
        pts_r = np.column_stack([np.cos(angles_r), np.sin(angles_r)])
        inits.append(pts_r)
    
    # 8. Random in unit disk
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts_d = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts_d)
    
    return inits

def _radial_optimize(pts, beta, time_budget=10.0, start_time=None):
    """Optimize radii only (convexity-preserving), keeping angles fixed."""
    if start_time is None:
        start_time = time.time()
    
    # Extract radii from current points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    x0 = radii.copy()
    
    result = minimize(
        _neg_soft_min_radial,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        bounds=[(0.01, 3.0)] * N,
        options={
            'maxiter': 500,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxls': 50
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return pts, _score(pts)
    
    new_radii = np.abs(result.x)
    new_pts = _radial_to_pts(new_radii, angles)
    return new_pts, _score(new_pts)

def _lbfgsb_optimize(pts, beta, time_budget=10.0, start_time=None):
    """Full 2D L-BFGS-B optimization on the soft-min surrogate."""
    if start_time is None:
        start_time = time.time()
    
    x0 = pts.flatten().copy()
    
    result = minimize(
        _neg_soft_min_score,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        options={
            'maxiter': 500,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxls': 50
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return pts, _score(pts)
    
    return result.x.reshape(N, 2), _score(result.x.reshape(N, 2))

def _progressive_beta_optimize(pts, start_time=None):
    """Progressive beta annealing: radial first (fast), then full 2D."""
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Phase A: Radial optimization with increasing beta (fast, 13 dims)
    betas_radial = [20, 50, 100, 200, 400]
    for beta in betas_radial:
        if time.time() > start_time + TIME_LIMIT - 30:
            break
        remaining = TIME_LIMIT - 25 - (time.time() - start_time)
        if remaining < 5:
            break
        stage_time = min(remaining * 0.2, 10.0)
        
        pts_opt, score_opt = _radial_optimize(best_pts, beta, time_budget=stage_time, start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase B: Full 2D optimization with increasing beta (26 dims)
    betas_full = [50, 100, 200, 400, 800]
    for beta in betas_full:
        if time.time() > start_time + TIME_LIMIT - 20:
            break
        remaining = TIME_LIMIT - 15 - (time.time() - start_time)
        if remaining < 5:
            break
        stage_time = min(remaining * 0.25, 12.0)
        
        pts_opt, score_opt = _lbfgsb_optimize(best_pts, beta, time_budget=stage_time, start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    return best_pts, best_score

def _targeted_polish(pts, time_budget=15.0, start_time=None):
    """Targeted polish: identify worst triangles and optimize involved points."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    areas = _triangle_areas(best_pts)
    n_triangles = len(areas)
    k_worst = min(5, n_triangles)
    worst_indices = np.argsort(areas)[:k_worst]
    
    involved_points = set()
    for wi in worst_indices:
        idx = _TRIS[wi]
        involved_points.add(idx[0])
        involved_points.add(idx[1])
        involved_points.add(idx[2])
    
    for pt_idx in sorted(involved_points):
        if time.time() > deadline - 2:
            break
        
        fixed_pts = best_pts.copy()
        x0 = fixed_pts[pt_idx].copy()
        
        def obj_single(x, idx=pt_idx, fp=fixed_pts):
            trial = fp.copy()
            trial[idx] = x
            return -_score(trial)
        
        res = minimize(
            obj_single,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if time.time() > deadline - 1:
            break
        
        if -res.fun > best_score + 1e-12:
            best_pts[pt_idx] = res.x
            best_score = -res.fun
    
    return best_pts, best_score

def _simulated_annealing(pts, time_budget=20.0, start_time=None):
    """Simulated annealing to escape local minima."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    
    T_start = 0.05
    T_end = 0.001
    n_steps = 1500
    
    rng = np.random.RandomState(777)
    
    for step in range(n_steps):
        if time.time() > deadline - 2:
            break
        
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        trial_pts = current_pts.copy()
        n_perturb = rng.randint(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        for i in idx:
            trial_pts[i] += T * rng.randn(2)
        
        trial_score = _score(trial_pts)
        
        if trial_score > current_score:
            current_pts = trial_pts
            current_score = trial_score
            if trial_score > best_score:
                best_score = trial_score
                best_pts = trial_pts.copy()
        else:
            delta = current_score - trial_score
            if delta > 0:
                accept_prob = np.exp(-delta / (T + 1e-15))
                if rng.random() < accept_prob:
                    current_pts = trial_pts
                    current_score = trial_score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -1
    
    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    # Phase 1: Progressive beta optimization (radial + full 2D) on each good init
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        s_init = _score(init)
        if s_init < best_score * 0.4:
            continue
        
        remaining = deadline - 30 - time.time()
        if remaining < 10:
            break
        
        pts_opt, score_opt = _progressive_beta_optimize(init, start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 2: Simulated annealing from best result
    remaining = deadline - 25 - time.time()
    if remaining > 10:
        best_pts, best_score = _simulated_annealing(best_pts, time_budget=min(remaining, 18.0), start_time=start_time)
    
    # Phase 3: Radial optimization on the SA result (fast refinement)
    remaining = deadline - 15 - time.time()
    if remaining > 5:
        for beta in [200, 500, 1000]:
            if time.time() > deadline - 12:
                break
            pts_opt, score_opt = _radial_optimize(best_pts, beta, time_budget=min(remaining * 0.3, 5.0), start_time=start_time)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
    
    # Phase 4: Full 2D L-BFGS-B with high beta
    remaining = deadline - 10 - time.time()
    if remaining > 3:
        for beta in [500, 1000, 2000]:
            if time.time() > deadline - 7:
                break
            pts_opt, score_opt = _lbfgsb_optimize(best_pts, beta, time_budget=min(remaining * 0.3, 5.0), start_time=start_time)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
    
    # Phase 5: Targeted polish on worst triangles
    remaining = deadline - 5 - time.time()
    if remaining > 3:
        best_pts, best_score = _targeted_polish(best_pts, time_budget=min(remaining, 4.0), start_time=start_time)
    
    # Phase 6: Final high-beta L-BFGS-B
    remaining = deadline - 2 - time.time()
    if remaining > 1.5:
        pts_opt, score_opt = _lbfgsb_optimize(best_pts, beta=2000, time_budget=min(remaining, 1.5), start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    return best_pts

# EVOLVE-BLOCK-END
