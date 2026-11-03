"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
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
    """Soft-min surrogate: -log-sum-exp(-beta * areas) / hull_area."""
    try:
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
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
    """Generate multiple initial configurations."""
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
    
    # 3. Halton on circle (angular Halton)
    angles_halton = 2 * np.pi * _halton_sequence(N, 2)
    radii = 0.5 + 0.5 * _halton_sequence(N, 3)
    pts_halton_circle = np.column_stack([radii * np.cos(angles_halton), radii * np.sin(angles_halton)])
    inits.append(pts_halton_circle)
    
    # 4. Grid-like: mix of boundary and interior
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
    
    # 5. Another mixed: 10 boundary, 3 interior
    angles_b = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    b_pts = np.column_stack([np.cos(angles_b), np.sin(angles_b)])
    i_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 6
    i_radii = [0.4, 0.6, 0.4]
    i_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(i_radii, i_angles)],
        [r * np.sin(a) for r, a in zip(i_radii, i_angles)]
    ])
    pts_mixed2 = np.vstack([b_pts, i_pts])
    inits.append(pts_mixed2)
    
    # 6. Random with seed
    rng = np.random.RandomState(42)
    pts_rand = rng.random((N, 2))
    inits.append(pts_rand)
    
    # 7. Regular polygon with one center point and 12 on circle
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    pts_center = np.vstack([circle_12, center])
    inits.append(pts_center)
    
    # 8. Two concentric rings: 7 outer, 6 inner
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    inner_pts = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    pts_rings = np.vstack([outer_pts, inner_pts])
    inits.append(pts_rings)
    
    # 9. Fibonacci lattice on disk
    golden_angle = np.pi * (3 - np.sqrt(5))
    radii_fib = np.sqrt(np.arange(1, N + 1) / N)
    angles_fib = golden_angle * np.arange(N)
    pts_fib = np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)])
    inits.append(pts_fib)
    
    # 10. Grid pattern
    xs = np.linspace(-0.8, 0.8, 4)
    ys = np.linspace(-0.8, 0.8, 3)
    grid_pts = np.array([[x, y] for x in xs for y in ys])
    if len(grid_pts) >= N:
        inits.append(grid_pts[:N])
    
    return inits

def _nelder_mead_polish(pts, time_limit=20.0, start_time=None):
    """Nelder-Mead simplex optimization to maximize min triangle area / hull area."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_limit
    
    def objective(x):
        """Minimize negative soft-min score for Nelder-Mead."""
        if time.time() > deadline:
            return 1e10
        pts = x.reshape(N, 2)
        return -_soft_min_score(pts, beta=30.0)
    
    # Initial simplex: perturb each point slightly
    x0 = pts.flatten()
    options = {
        'maxiter': 5000,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'disp': False
    }
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=options)
        if result.x is not None:
            best_pts = result.x.reshape(N, 2)
            best_score = _score(best_pts)
            return best_pts, best_score
    except Exception:
        pass
    
    return pts, _score(pts)

def _coordinate_descent_polish(pts, max_iter=200, time_limit=20.0, start_time=None):
    """Coordinate descent polish: optimize each point's x,y separately."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_limit
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Adaptive step sizes
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try different step sizes for x and y
            for step in step_sizes:
                # Optimize x coordinate of point i
                for dx in [step, -step]:
                    trial = best_pts.copy()
                    trial[i, 0] += dx
                    s = _score(trial)
                    if s > best_score + 1e-12:
                        best_pts = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
                
                # Optimize y coordinate of point i
                if not improved:
                    for dy in [step, -step]:
                        trial = best_pts.copy()
                        trial[i, 1] += dy
                        s = _score(trial)
                        if s > best_score + 1e-12:
                            best_pts = trial
                            best_score = s
                            improved = True
                            break
                if improved:
                    break
            
            if time.time() > deadline:
                break
        
        if not improved:
            break
    
    return best_pts, best_score

def _differential_evolution_search(bounds, x0_list, time_budget, start_time):
    """Run differential evolution to maximize soft-min score."""
    deadline = start_time + time_budget
    
    results = []
    
    # First run: moderate settings
    popsize = 15
    max_iter = 80
    tol = 1e-6
    
    def timeout_callback(xk, convergence):
        return time.time() > deadline - 5
    
    try:
        res = differential_evolution(
            _neg_soft_min_score,
            bounds,
            x0=None,
            popsize=popsize,
            maxiter=max_iter,
            tol=tol,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            callback=timeout_callback,
            init='latinhypercube'
        )
        if res.success or res.x is not None:
            best_x = res.x
            best_s = -res.fun
            results.append((best_x, best_s))
    except Exception:
        pass
    
    # Second run: different seed and settings if time remains
    remaining = deadline - 5 - time.time()
    if remaining > 20:
        try:
            res2 = differential_evolution(
                _neg_soft_min_score,
                bounds,
                x0=None,
                popsize=12,
                maxiter=60,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=123,
                polish=True,
                callback=timeout_callback,
                init='latinhypercube'
            )
            if res2.success or res2.x is not None:
                best_x2 = res2.x
                best_s2 = -res2.fun
                results.append((best_x2, best_s2))
        except Exception:
            pass
    
    # Select best result
    if results:
        results.sort(key=lambda r: r[1], reverse=True)
        return results[0][0]
    
    return None

def _local_search(pts, start_time, time_budget=15.0):
    """Local search with random perturbations and acceptance."""
    deadline = start_time + time_budget
    best_pts = pts.copy()
    best_score = _score(best_pts)
    rng = np.random.RandomState(int(time.time() * 1000) % 10000)
    
    current_temp = 0.1
    for iteration in range(2000):
        if time.time() > deadline - 1:
            break
        
        # Random perturbation
        trial = best_pts + rng.normal(0, current_temp, size=(N, 2))
        trial_score = _score(trial)
        
        if trial_score > best_score:
            best_pts = trial
            best_score = trial_score
            current_temp *= 0.95
        else:
            # Accept with probability based on score difference (simulated annealing)
            delta = best_score - trial_score
            if delta < 0.01 and rng.random() < np.exp(-delta / max(current_temp, 1e-6)):
                best_pts = trial
                current_temp *= 0.98
            else:
                current_temp *= 0.99
    
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
    
    # Phase 1: Differential evolution for global search
    remaining = deadline - 40 - time.time()
    if remaining > 30:
        bounds = [(-1.0, 1.0)] * (2 * N)
        de_result = _differential_evolution_search(bounds, inits, remaining, start_time)
        if de_result is not None:
            de_pts = de_result.reshape(N, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts
    
    # Phase 2: Polish each good initialization with Nelder-Mead
    for i, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
        s_init = _score(init)
        if s_init < best_score * 0.7:
            continue
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        # Use Nelder-Mead for better local optimization
        pts_opt, score_opt = _nelder_mead_polish(init, time_limit=min(remaining, 15.0), start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 3: Local search with simulated annealing
    remaining = deadline - 15 - time.time()
    if remaining > 10:
        best_pts, best_score = _local_search(best_pts, start_time, time_budget=remaining)
    
    # Phase 4: Final polish on the best result with Nelder-Mead
    remaining = deadline - 5 - time.time()
    if remaining > 3:
        best_pts, best_score = _nelder_mead_polish(best_pts, time_limit=remaining, start_time=start_time)
    
    # Phase 5: Final coordinate descent for fine-tuning
    remaining = deadline - 2 - time.time()
    if remaining > 1:
        best_pts, best_score = _coordinate_descent_polish(best_pts, max_iter=100, time_limit=remaining, start_time=start_time)
    
    return best_pts

# EVOLVE-BLOCK-END
