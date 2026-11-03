"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Uses smooth log-sum-exp soft-min surrogate + L-BFGS-B gradient-based optimization
with progressive beta annealing and concentric ring initializations."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        rest = pts[i+1:]
        m = len(rest)
        v1 = rest[:, np.newaxis, :] - pi
        v2 = rest[np.newaxis, :, :] - pi
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        idx = np.triu_indices(m, k=1)
        areas.append(np.abs(cross[idx]))
    areas = np.concatenate(areas)
    return 0.5 * areas

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
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min_score(pts, beta=50.0):
    """Soft-min surrogate: smooth approximation of min(areas)/hull_area."""
    try:
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        # Soft min: -log(sum(exp(-beta * a))) / beta
        # Use log-sum-exp trick for numerical stability
        max_a = np.max(areas)
        shifted = areas - max_a
        log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * max_a)
        soft_min = -log_sum_exp / beta
        return soft_min / hull_area
    except Exception:
        return 0.0

def _neg_soft_min_score(x, beta=50.0):
    """Negative soft-min score for minimization (for scipy)."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _neg_score(x):
    """Negative exact score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0,1)."""
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

def _generate_initializations():
    """Generate multiple diverse initial configurations including concentric rings."""
    inits = []
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_circle)
    
    # 2. Halton sequence in unit square
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
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
    
    # 6. Halton on circle (angular Halton)
    angles_halton = 2 * np.pi * _halton_sequence(N, 2)
    radii = 0.5 + 0.5 * _halton_sequence(N, 3)
    pts_halton_circle = np.column_stack([radii * np.cos(angles_halton), radii * np.sin(angles_halton)])
    inits.append(pts_halton_circle)
    
    # 7. Regular polygon with center point and 12 on circle
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    pts_center = np.vstack([circle_12, center])
    inits.append(pts_center)
    
    # 8. Random with seed
    rng = np.random.RandomState(42)
    pts_rand = rng.random((N, 2))
    inits.append(pts_rand)
    
    # 9. Another random with different seed
    rng2 = np.random.RandomState(123)
    pts_rand2 = rng2.random((N, 2))
    inits.append(pts_rand2)
    
    # 10. Grid-like: mix of boundary and interior
    angles_boundary = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    boundary_pts = np.column_stack([np.cos(angles_boundary), np.sin(angles_boundary)])
    interior_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    interior_radii = [0.3, 0.5, 0.3, 0.5]
    interior_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(interior_radii, interior_angles)],
        [r * np.sin(a) for r, a in zip(interior_radii, interior_angles)]
    ])
    pts_mixed = np.vstack([boundary_pts, interior_pts])
    inits.append(pts_mixed)
    
    return inits

def _l_bfgs_b_optimize(pts, beta, time_budget=10.0, start_time=None):
    """L-BFGS-B optimization on the soft-min surrogate."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = pts.flatten().copy()
    
    result = minimize(
        _neg_soft_min_score,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        options={
            'maxiter': 500,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 50
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return pts, _score(pts)
    
    return result.x.reshape(N, 2), _score(result.x.reshape(N, 2))

def _progressive_beta_optimize(pts, start_time=None):
    """Progressive beta annealing: start with low beta for smooth landscape,
    increase beta for sharper min approximation."""
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Progressive beta values: start smooth, get sharper
    betas = [10, 20, 40, 80, 150, 200, 300]
    
    for beta in betas:
        if time.time() > start_time + TIME_LIMIT - 20:
            break
        
        # Check time budget for this stage
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - 15 - elapsed
        if remaining < 5:
            break
        
        # Allocate time proportional to remaining
        stage_time = min(remaining * 0.25, 12.0)
        
        pts_opt, score_opt = _l_bfgs_b_optimize(best_pts, beta, time_budget=stage_time, start_time=start_time)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    return best_pts, best_score

def _nelder_mead_polish(pts, time_budget=10.0, start_time=None):
    """Nelder-Mead optimization on the full 2N-dimensional space for final polish."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = pts.flatten().copy()
    
    result = minimize(
        _neg_score,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 1500,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return pts, _score(pts)
    
    return result.x.reshape(N, 2), -result.fun

def _targeted_polish(pts, time_budget=15.0, start_time=None):
    """Targeted polish: identify worst triangles and optimize points involved."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Find the worst (smallest area) triangles
    areas = _all_triangle_areas(best_pts)
    min_area = np.min(areas)
    
    # Get indices of the 5 smallest triangles
    n_triangles = len(areas)
    k_worst = min(5, n_triangles)
    worst_indices = np.argsort(areas)[:k_worst]
    
    # Collect all points involved in worst triangles
    n = N
    
    def tri_idx_to_pts(tri_idx):
        """Convert triangle index to point indices (i,j,k with i<j<k)."""
        count = 0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if count == tri_idx:
                        return (i, j, k)
                    count += 1
        return (0, 1, 2)
    
    involved_points = set()
    for wi in worst_indices:
        i, j, k = tri_idx_to_pts(wi)
        involved_points.add(i)
        involved_points.add(j)
        involved_points.add(k)
    
    # Optimize each involved point using Nelder-Mead on just that point's coordinates
    for pt_idx in sorted(involved_points):
        if time.time() > deadline - 2:
            break
        
        # Create a function that optimizes only point pt_idx
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
                'maxiter': 400,
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

def _simulated_annealing(pts, time_budget=30.0, start_time=None):
    """Simulated annealing to escape local minima."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    
    # Initial temperature
    T_start = 0.05
    T_end = 0.001
    n_steps = 2000
    
    rng = np.random.RandomState(777)
    
    for step in range(n_steps):
        if time.time() > deadline - 2:
            break
        
        # Annealing schedule
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        trial_pts = current_pts.copy()
        # Randomly perturb 1-3 points
        n_perturb = rng.randint(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        for i in idx:
            trial_pts[i] += T * rng.randn(2)
        
        trial_score = _score(trial_pts)
        
        # Accept or reject
        if trial_score > current_score:
            current_pts = trial_pts
            current_score = trial_score
            if trial_score > best_score:
                best_score = trial_score
                best_pts = trial_pts.copy()
        else:
            # Metropolis criterion
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
    
    # Phase 1: Progressive beta L-BFGS-B optimization on each good initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        s_init = _score(init)
        if s_init < best_score * 0.5:
            continue
        
        remaining = deadline - 25 - time.time()
        if remaining < 10:
            break
        
        # Use progressive beta optimization
        pts_opt, score_opt = _progressive_beta_optimize(init, start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 2: Simulated annealing from best result
    remaining = deadline - 20 - time.time()
    if remaining > 10:
        best_pts, best_score = _simulated_annealing(best_pts, time_budget=min(remaining, 20.0), start_time=start_time)
    
    # Phase 3: Targeted polish on the best result
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        best_pts, best_score = _targeted_polish(best_pts, time_budget=min(remaining, 10.0), start_time=start_time)
    
    # Phase 4: Final L-BFGS-B with high beta
    remaining = deadline - 5 - time.time()
    if remaining > 3:
        pts_opt, score_opt = _l_bfgs_b_optimize(best_pts, beta=300, time_budget=min(remaining, 5.0), start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 5: Final Nelder-Mead polish for precision
    remaining = deadline - 2 - time.time()
    if remaining > 2:
        best_pts, best_score = _nelder_mead_polish(best_pts, time_budget=min(remaining, 3.0), start_time=start_time)
    
    return best_pts

# EVOLVE-BLOCK-END
