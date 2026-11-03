"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
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
    """Soft-min surrogate: -log-sum-exp(-beta * areas) / hull_area.
    This is a smooth approximation of min(areas)/hull_area that provides
    gradient information for optimization."""
    try:
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        # Soft min: (1/beta) * log(sum(exp(-beta * a)))
        # We want to maximize min, so we minimize -soft_min
        # soft_min = -log(sum(exp(-beta * a))) / beta
        # To avoid overflow, use log-sum-exp trick
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
    
    return inits

def _coordinate_descent_polish(pts, max_iter=200, time_limit=20.0, start_time=None):
    """Coordinate descent polish: optimize each point's x,y separately."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_limit
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Optimize x coordinate of point i
            for dx in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02]:
                trial = best_pts.copy()
                trial[i, 0] += dx
                s = _score(trial)
                if s > best_score + 1e-12:
                    best_pts = trial
                    best_score = s
                    improved = True
                    break
            
            # Optimize y coordinate of point i
            for dy in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02]:
                trial = best_pts.copy()
                trial[i, 1] += dy
                s = _score(trial)
                if s > best_score + 1e-12:
                    best_pts = trial
                    best_score = s
                    improved = True
                    break
            
            if time.time() > deadline:
                break
        
        if not improved:
            # Try smaller steps
            for dx in [0.0005, -0.0005, 0.002, -0.002]:
                for i in range(N):
                    trial = best_pts.copy()
                    trial[i, 0] += dx
                    s = _score(trial)
                    if s > best_score + 1e-12:
                        best_pts = trial
                        best_score = s
                        improved = True
                        break
                    trial = best_pts.copy()
                    trial[i, 1] += dx
                    s = _score(trial)
                    if s > best_score + 1e-12:
                        best_pts = trial
                        best_score = s
                        improved = True
                        break
                if improved or time.time() > deadline:
                    break
            if not improved:
                break
    
    return best_pts, best_score

def _simulated_annealing_search(pts0, time_budget, start_time, seed=42):
    """Simulated annealing on the true score function.
    
    Structural change: uses the exact min-triangle/hull-area objective directly,
    with geometric perturbation moves (rotate, translate, scale individual points,
    swap pairs) rather than a soft-min surrogate. This explores the landscape
    differently than differential evolution.
    """
    deadline = start_time + time_budget
    rng = np.random.RandomState(seed)
    
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    curr_pts = pts0.copy()
    curr_score = best_score
    
    # Adaptive temperature
    T0 = 0.05
    Tend = 0.0001
    n_steps_target = max(1000, int(time_budget * 200))
    cooling = (Tend / T0) ** (1.0 / max(1, n_steps_target - 1))
    
    step_size = 0.05
    
    for step in range(n_steps_target):
        if time.time() > deadline - 2:
            break
        
        T = T0 * (cooling ** step)
        
        # Generate a perturbation
        move_type = rng.randint(0, 5)
        trial = curr_pts.copy()
        
        if move_type == 0:
            # Random single point nudge
            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            trial[i, 0] += dx
            trial[i, 1] += dy
        elif move_type == 1:
            # Rotate a point around centroid
            i = rng.randint(N)
            cx = np.mean(curr_pts[:, 0])
            cy = np.mean(curr_pts[:, 1])
            theta = rng.uniform(-step_size, step_size)
            vx = curr_pts[i, 0] - cx
            vy = curr_pts[i, 1] - cy
            trial[i, 0] = cx + vx * np.cos(theta) - vy * np.sin(theta)
            trial[i, 1] = cy + vx * np.sin(theta) + vy * np.cos(theta)
        elif move_type == 2:
            # Scale a point's distance from centroid
            i = rng.randint(N)
            cx = np.mean(curr_pts[:, 0])
            cy = np.mean(curr_pts[:, 1])
            factor = 1.0 + rng.uniform(-step_size, step_size)
            trial[i, 0] = cx + (curr_pts[i, 0] - cx) * factor
            trial[i, 1] = cy + (curr_pts[i, 1] - cy) * factor
        elif move_type == 3:
            # Swap two points partially
            i, j = rng.choice(N, 2, replace=False)
            t = rng.uniform(0.1, 0.3)
            trial[i, :] = (1-t) * curr_pts[i, :] + t * curr_pts[j, :]
            trial[j, :] = (1-t) * curr_pts[j, :] + t * curr_pts[i, :]
        else:
            # Random pair nudge in opposite directions
            i, j = rng.choice(N, 2, replace=False)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            trial[i, 0] += dx
            trial[i, 1] += dy
            trial[j, 0] -= dx
            trial[j, 1] -= dy
        
        trial_score = _score(trial)
        
        # Accept if better or with probability exp(-delta/T)
        if trial_score >= curr_score:
            curr_pts = trial
            curr_score = trial_score
            if curr_score > best_score:
                best_score = curr_score
                best_pts = trial.copy()
        else:
            delta = curr_score - trial_score
            if delta > 0:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    curr_pts = trial
                    curr_score = trial_score
        
        # Adaptive step size
        if step % 500 == 0:
            # Shrink step size slowly
            step_size *= 0.95
    
    return best_pts, best_score


def _targeted_repair(pts, time_budget, start_time):
    """Identify worst triangles and try to increase their areas by moving vertices."""
    deadline = start_time + time_budget
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for iteration in range(50):
        if time.time() > deadline - 2:
            break
        
        areas = _all_triangle_areas(best_pts)
        min_idx = np.argmin(areas)
        
        # Find which triple has the minimum area
        # Reconstruct triple from index
        n = N
        count = 0
        triple = None
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if count == min_idx:
                        triple = (i, j, k)
                        break
                    count += 1
                if triple:
                    break
            if triple:
                break
        
        if triple is None:
            break
        
        i, j, k = triple
        pi, pj, pk = best_pts[i], best_pts[j], best_pts[k]
        
        # Try moving each vertex slightly away from the opposite side
        for vi, vj, vk in [(pi, pj, pk), (pj, pk, pi), (pk, pi, pj)]:
            # Direction to move: away from the line formed by other two
            # For vertex i, move away from line jk
            pass  # Simplified: just try random small moves
        
        # Try random small moves on the three vertices
        rng = np.random.RandomState(iteration * 1000)
        for _ in range(20):
            trial = best_pts.copy()
            for idx in [i, j, k]:
                trial[idx, 0] += rng.uniform(-0.01, 0.01)
                trial[idx, 1] += rng.uniform(-0.01, 0.01)
            s = _score(trial)
            if s > best_score + 1e-12:
                best_pts = trial
                best_score = s
                break
        
        # Also try coordinate descent on the three vertices
        for idx in [i, j, k]:
            for coord in [0, 1]:
                for delta in [0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                    trial = best_pts.copy()
                    trial[idx, coord] += delta
                    s = _score(trial)
                    if s > best_score + 1e-12:
                        best_pts = trial
                        best_score = s
                        break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -1
    
    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    # Phase 1: Simulated annealing from multiple starting points
    sa_configs = [
        (inits[0], 42),   # regular polygon
        (inits[1], 123),  # Halton
        (inits[3], 456),  # mixed boundary/interior
        (inits[6], 789),  # center + circle
    ]
    
    for init, seed in sa_configs:
        remaining = deadline - 25 - time.time()
        if remaining < 5:
            break
        sa_pts, sa_score = _simulated_annealing_search(init, remaining * 0.7, start_time, seed=seed)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
    
    # Phase 2: Coordinate descent polish on best
    remaining = deadline - 15 - time.time()
    if remaining > 5:
        pts_opt, score_opt = _coordinate_descent_polish(best_pts, max_iter=200, time_limit=min(remaining, 12.0), start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 3: Targeted repair of worst triangles
    remaining = deadline - 8 - time.time()
    if remaining > 3:
        pts_repair, score_repair = _targeted_repair(best_pts, remaining, start_time)
        if score_repair > best_score:
            best_score = score_repair
            best_pts = pts_repair
    
    # Phase 4: Final coordinate descent polish
    remaining = deadline - 2 - time.time()
    if remaining > 2:
        best_pts, best_score = _coordinate_descent_polish(best_pts, max_iter=300, time_limit=remaining, start_time=start_time)
    
    return best_pts

# EVOLVE-BLOCK-END
