"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)  # 165 for N=11


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _push_apart_worst(pts, step_size=0.01):
    """
    Targeted local move: find the worst (min-area) triangle and push its
    three vertices apart from each other. This is a structural local search
    that directly addresses the bottleneck triangle.
    """
    areas = _all_areas(pts)
    min_idx = np.argmin(areas)
    i, j, k = _TRIPLETS[min_idx]
    
    # Get the three points
    pi = pts[i].copy()
    pj = pts[j].copy()
    pk = pts[k].copy()
    
    # Compute centroid of the worst triangle
    centroid = (pi + pj + pk) / 3.0
    
    # Push each vertex away from the centroid
    new_pts = pts.copy()
    for idx, pt in [(i, pi), (j, pj), (k, pk)]:
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            new_pts[idx] = pt + step_size * direction
    
    # Also push the three vertices away from each other pairwise
    for a, b in [(i, j), (j, k), (i, k)]:
        d = new_pts[a] - new_pts[b]
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            d = d / norm
            new_pts[a] = new_pts[a] + 0.5 * step_size * d
            new_pts[b] = new_pts[b] - 0.5 * step_size * d
    
    new_pts = _project_to_triangle(new_pts)
    return new_pts


def _random_local_move(pts, rng, step_size=0.02):
    """Random perturbation of one or two points."""
    new_pts = pts.copy()
    # Pick 1 or 2 random points to move
    n_move = 1 if rng.random() < 0.5 else 2
    indices = rng.choice(N, size=n_move, replace=False)
    for idx in indices:
        new_pts[idx] = new_pts[idx] + rng.normal(0, step_size, size=2)
    new_pts = _project_to_triangle(new_pts)
    return new_pts


def _simulated_annealing(start_pts, rng, start_time, deadline):
    """
    Simulated annealing with targeted local refinement.
    Global exploration via random moves, local exploitation via push-apart.
    """
    current = start_pts.copy()
    current_area = _min_triangle_area(current)
    
    best = current.copy()
    best_area = current_area
    
    # Annealing schedule
    T0 = 0.01  # initial temperature
    Tend = 0.0001  # final temperature
    n_steps = 30000
    
    # Time-based step limit
    time_per_step = (deadline - start_time) / n_steps
    
    for step in range(n_steps):
        if (time.time() - start_time) > deadline:
            break
        
        # Temperature schedule (exponential decay)
        T = T0 * (Tend / T0) ** (step / n_steps)
        
        # Choose move type: 70% random local, 30% push-apart
        if rng.random() < 0.7:
            # Random local move with adaptive step size
            step_size = 0.01 + 0.02 * (T / T0)
            candidate = _random_local_move(current, rng, step_size)
        else:
            # Push apart the worst triangle
            step_size = 0.005 + 0.01 * (T / T0)
            candidate = _push_apart_worst(current, step_size)
        
        cand_area = _min_triangle_area(candidate)
        
        # Metropolis acceptance
        delta = cand_area - current_area
        if delta > 0 or rng.random() < np.exp(delta / T):
            current = candidate
            current_area = cand_area
            
            if current_area > best_area:
                best_area = current_area
                best = current.copy()
    
    return best, best_area


def _iterative_worst_triangle_refine(pts, rng, max_iters=500):
    """
    Iteratively refine by always pushing apart the current worst triangle.
    With small random perturbations to escape local optima.
    """
    current = pts.copy()
    current_area = _min_triangle_area(current)
    best = current.copy()
    best_area = current_area
    
    step_size = 0.005
    
    for it in range(max_iters):
        candidate = _push_apart_worst(current, step_size)
        cand_area = _min_triangle_area(candidate)
        
        if cand_area > current_area:
            current = candidate
            current_area = cand_area
            if current_area > best_area:
                best_area = current_area
                best = current.copy()
        else:
            # Stuck: add small random perturbation to the worst triangle's vertices
            areas = _all_areas(current)
            min_idx = np.argmin(areas)
            i, j, k = _TRIPLETS[min_idx]
            for idx in [i, j, k]:
                current[idx] = current[idx] + rng.normal(0, 0.003, size=2)
            current = _project_to_triangle(current)
            current_area = _min_triangle_area(current)
        
        # Gradually reduce step size
        if it % 50 == 0:
            step_size *= 0.95
    
    return best, best_area


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Multiple diverse initial configurations
    inits = []
    
    # Init 1: Symmetric distribution
    init1 = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [0.5, S3H/3],        # center-ish
        [0.333, S3H/6],      # lower left
        [0.667, S3H/6],      # lower right
        [0.5, 2*S3H/3],      # upper
        [0.2, 0.2],
    ], dtype=float)
    inits.append(init1)
    
    # Init 2: Grid-like distribution
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, S3H/2],
        [0.75, S3H/2],
        [0.5, S3H/2],
        [0.5, S3H/6],
        [0.5, S3H*2/3],
        [0.5, 0.0],
    ], dtype=float)
    inits.append(init2)
    
    # Init 3: Random but spread
    init3 = _project_to_triangle(rng.random((N, 2)) * np.array([1.0, S3H]))
    inits.append(init3)
    
    # Evaluate all inits
    for init in inits:
        _update_best(init)
    
    # Phase A: Simulated Annealing from best init
    if _elapsed_ok(30):
        sa_deadline = start_time + 80
        sa_best, sa_area = _simulated_annealing(best_pts, rng, start_time, sa_deadline)
        _update_best(sa_best)
    
    # Phase B: Iterative worst-triangle refinement from best so far
    if best_pts is not None and _elapsed_ok(20):
        refine_deadline = start_time + 100
        # Run multiple refinement passes
        for pass_num in range(3):
            if not _elapsed_ok(10):
                break
            # Start from current best or a perturbed version
            if pass_num == 0:
                start = best_pts.copy()
            else:
                start = best_pts + rng.normal(0, 0.03, size=(N, 2))
                start = _project_to_triangle(start)
            
            time_left = DEADLINE - (time.time() - start_time) - 10
            if time_left <= 0:
                break
            
            # Estimate max iterations based on time
            # Each iteration takes ~0.1ms, so ~10000 iters in 1 second
            max_iters = int(min(2000, max(100, time_left * 5000)))
            
            ref_best, ref_area = _iterative_worst_triangle_refine(
                start, rng, max_iters=max_iters
            )
            _update_best(ref_best)
    
    # Phase C: Final targeted refinement on the absolute best
    if best_pts is not None and _elapsed_ok(5):
        # Very fine-grained push-apart on the worst triangle
        fine_best = best_pts.copy()
        fine_area = _min_triangle_area(fine_best)
        step_size = 0.001
        
        for _ in range(200):
            if not _elapsed_ok(2):
                break
            candidate = _push_apart_worst(fine_best, step_size)
            cand_area = _min_triangle_area(candidate)
            if cand_area > fine_area:
                fine_best = candidate
                fine_area = cand_area
            else:
                # Tiny random nudge
                areas = _all_areas(fine_best)
                min_idx = np.argmin(areas)
                i, j, k = _TRIPLETS[min_idx]
                for idx in [i, j, k]:
                    fine_best[idx] = fine_best[idx] + rng.normal(0, 0.0005, size=2)
                fine_best = _project_to_triangle(fine_best)
                fine_area = _min_triangle_area(fine_best)
        
        _update_best(fine_best)

    if best_pts is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u + 0.5 * v
        y = v * S3H
        return np.column_stack([x, y])

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
