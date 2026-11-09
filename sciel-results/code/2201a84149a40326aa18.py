# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull (vectorized)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Calculate all triangle areas efficiently using combinations
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate using log-sum-exp for soft-min of triangle areas.
    Returns a differentiable-ish proxy that is smoother than hard min.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min: -T * log(sum(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    return soft_min_area / A_hull

def nelder_mead_optimize(points, time_limit, temperature=0.01, max_iter=5000):
    """
    Nelder-Mead optimization on the soft-min surrogate, then refine with hard min.
    """
    points = points.copy()
    n = len(points)
    flat = points.flatten()
    
    # Create initial simplex
    simplex = [flat.copy()]
    step = 0.1
    for i in range(2 * n):
        p = flat.copy()
        p[i] += step
        simplex.append(p)
    
    def objective(flat_arr):
        pts = flat_arr.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        return -soft_min_score(pts, temperature)
    
    best_flat = flat.copy()
    best_val = objective(flat)
    
    # Nelder-Mead iterations
    for iteration in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Sort simplex
        vals = [objective(s) for s in simplex]
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        
        best_val = vals[0]
        if best_val < objective(best_flat):
            best_flat = simplex[0].copy()
        
        # Centroid of all but worst
        worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        reflected = centroid + alpha * (centroid - worst)
        ref_val = objective(reflected)
        
        if ref_val < vals[0]:
            # Expansion
            gamma = 2.0
            expanded = centroid + gamma * (centroid - worst)
            exp_val = objective(expanded)
            if exp_val < ref_val:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        elif ref_val < vals[-2]:
            simplex[-1] = reflected
        else:
            # Contraction
            rho = 0.5
            contracted = centroid + rho * (worst - centroid)
            con_val = objective(contracted)
            if con_val < vals[-1]:
                simplex[-1] = contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    # Convert back to points
    best_points = best_flat.reshape(n, 2)
    
    # Polish with hard min via local search
    hard_score = compute_score(best_points)
    
    return best_points, hard_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    
    Uses a combination of structured initial configurations and local search
    with random perturbations, inspired by high-scoring solutions.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Regular 12-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    pts1 = np.vstack([np.array([0.0, 0.0]), pts1])
    candidates.append(pts1)
    
    # Candidate 2: Two concentric hexagons + center
    # Hexagon 1: radius 0.5, angles 0, 60, ...
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts_h1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    # Hexagon 2: radius 1.0, angles 30, 90, ...
    angles2 = angles1 + np.pi/6
    pts_h2 = np.column_stack([1.0 * np.cos(angles2), 1.0 * np.sin(angles2)])
    pts2 = np.vstack([np.array([0.0, 0.0]), pts_h1, pts_h2])
    candidates.append(pts2)
    
    # Candidate 3: Two concentric polygons (7 and 6)
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    candidates.append(make_concentric_polygons(7, 6, 1.0, 0.5))
    candidates.append(make_concentric_polygons(6, 7, 1.0, 0.6))
    
    # Candidate 4: Fibonacci-like distribution
    def make_fibonacci_2d():
        n = 13
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n) / golden_ratio
        radii = np.sqrt(np.arange(n) / n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    candidates.append(make_fibonacci_2d())
    
    # Candidate 5: Grid-like arrangement
    def make_grid_arrangement():
        # 4x3 grid with one extra point
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 2, 3])
        # Take 13 points from a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < 13:
                    points.append([i, j])
        points = np.array(points)
        # Center and scale
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        return points
    
    candidates.append(make_grid_arrangement())
    
    # Add random perturbations to structured configs
    rng = np.random.default_rng(12345)
    for base in candidates[:3]:
        perturb = rng.normal(0, 0.05, size=base.shape)
        candidates.append(base + perturb)
    
    # Add some fully random starts
    for _ in range(5):
        pts_rand = rng.random((13, 2)) * 2 - 1
        candidates.append(pts_rand)
    
    # Candidate: Ring with jittered inner points
    def make_ring_inner():
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
        return np.vstack([outer, inner])
    
    candidates.append(make_ring_inner())
    candidates.append(make_ring_inner() + rng.normal(0, 0.03, size=(13,2)))
    
    # Candidate: Alternating radii pattern
    def make_alternating():
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        radii = np.ones(13)
        radii[::2] = 1.0
        radii[1::2] = 0.6
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
    
    candidates.append(make_alternating())
    candidates.append(make_alternating() + rng.normal(0, 0.03, size=(13,2)))
    
    # Local search function with Nelder-Mead polish
    def local_search(points, time_limit):
        points = points.copy()
        current_score = compute_score(points)
        best_score_local = current_score
        best_points_local = points.copy()
        
        step_size = 0.1
        no_improve_count = 0
        rng = np.random.default_rng(seed=int(time.time() * 1000) % 100000)
        
        # Phase 1: Random perturbation local search
        while time.time() < time_limit * 0.5:
            improved = False
            for idx in range(13):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                
                new_points = points.copy()
                new_points[idx, 0] += dx
                new_points[idx, 1] += dy
                
                new_score = compute_score(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_points_local = points.copy()
                else:
                    no_improve_count += 1
            
            if not improved:
                step_size *= 0.95
                if step_size < 1e-6:
                    break
            else:
                pass
        
        # Phase 2: Nelder-Mead polish on the best found
        if time.time() < time_limit:
            nm_deadline = time_limit
            nm_pts, nm_score = nelder_mead_optimize(best_points_local, nm_deadline, temperature=0.005)
            if nm_score > best_score_local:
                best_score_local = nm_score
                best_points_local = nm_pts
        
        # Phase 3: Final fine-tuning with smaller steps
        if time.time() < time_limit - 5:
            step_size = 0.01
            current_score = best_score_local
            points = best_points_local.copy()
            for _ in range(200):
                if time.time() > time_limit - 5:
                    break
                improved = False
                for idx in range(13):
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)
                    
                    new_points = points.copy()
                    new_points[idx, 0] += dx
                    new_points[idx, 1] += dy
                    
                    new_score = compute_score(new_points)
                    
                    if new_score > current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_points_local = points.copy()
                    else:
                        pass
                
                if not improved:
                    step_size *= 0.8
                    if step_size < 1e-7:
                        break
        
        return best_points_local, best_score_local
    
    # Run local search on each candidate
    n_candidates = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline:
            break
            
        # Allocate time for this candidate
        # Distribute remaining time among remaining candidates
        remaining_time = deadline - time.time()
        remaining_candidates = n_candidates - i
        candidate_time = remaining_time / remaining_candidates
        candidate_deadline = min(deadline, time.time() + candidate_time)
        
        res_pts, res_score = local_search(init_pts, candidate_deadline)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
            
    # If we have time left, do a final polish on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        # Run another round of local search on the best found
        res_pts, res_score = local_search(best_points, deadline - 10.0)
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
    
    # Final fallback
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
