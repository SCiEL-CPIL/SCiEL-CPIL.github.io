"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

DEADLINE = 140.0  # safety margin before 150s kill


def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Area of convex hull of points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: use polygon area if all points are on convex position
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            return hull.area
        except Exception:
            # Last resort: shoelace formula on sorted points
            pts = points[np.argsort(points[:, 0])]
            x, y = pts[:, 0], pts[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _make_points_from_params(params):
    """
    Create points from parameters:
    params = [r1, r2, ..., rm, phase1, phase2, ..., phasem]
    where m is the number of shells, each shell is a regular polygon.
    Shell i has n_i points at radius r_i and rotation phase_i.
    The partition is fixed (e.g., [7, 6] or [5, 4, 4], etc.)
    """
    # Partition: we'll try several partitions
    # For now, use a fixed partition based on params length
    # params: [r1, phase1, r2, phase2, ...] for m shells
    m = len(params) // 2
    if m == 0:
        return None
    
    # Fixed partition for m shells
    # We'll use a global partition
    global _current_partition
    if _current_partition is None or len(_current_partition) != m:
        return None
    
    points = []
    for i in range(m):
        r = params[2 * i]
        phase = params[2 * i + 1]
        n_i = _current_partition[i]
        angles = 2 * np.pi * np.arange(n_i) / n_i + phase
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        for j in range(n_i):
            points.append([x[j], y[j]])
    
    return np.array(points)


def _objective(params, partition):
    """Objective function to minimize (negative score)."""
    global _current_partition
    _current_partition = partition
    
    points = _make_points_from_params(params)
    if points is None or len(points) != N:
        return 1e10
    
    # Check for degenerate points (too close together)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                return 1e10
    
    s = _score(points)
    return -s


def _optimize_partition(partition, time_limit):
    """Optimize a given partition of N points into shells."""
    global _current_partition
    _current_partition = partition
    
    m = len(partition)
    n_params = 2 * m  # radius and phase for each shell
    
    best_score = 0.0
    best_params = None
    
    # Try multiple random initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > time_limit:
            break
        
        # Random initial radii and phases
        if restart == 0:
            # Start with concentric shells
            radii = np.linspace(0.3, 1.0, m)
            phases = np.zeros(m)
        else:
            radii = 0.3 + 0.7 * np.random.rand(m)
            phases = 2 * np.pi * np.random.rand(m)
        
        params0 = np.zeros(n_params)
        for i in range(m):
            params0[2 * i] = radii[i]
            params0[2 * i + 1] = phases[i]
        
        try:
            result = minimize(
                _objective,
                params0,
                args=(partition,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < -best_score:  # We're minimizing -score
                best_score = -result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
    
    return best_score, best_params


def construct():
    """Place N points to maximize the minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Define possible partitions of 13 into m parts
    # Each part must be >= 3 for a regular polygon to make sense
    partitions = [
        [13],           # Single regular 13-gon
        [7, 6],         # Two shells
        [6, 7],         # Two shells (different order)
        [5, 4, 4],      # Three shells
        [5, 5, 3],      # Three shells
        [4, 4, 5],      # Three shells
        [4, 5, 4],      # Three shells
        [5, 3, 5],      # Three shells
        [3, 5, 5],      # Three shells
        [4, 4, 3, 2],   # Four shells (but 2-gon is degenerate, skip)
        [3, 3, 3, 4],   # Four shells
        [4, 3, 3, 3],   # Four shells
    ]
    
    # Filter out invalid partitions (all parts must be >= 3)
    valid_partitions = [p for p in partitions if all(x >= 3 for x in p) and sum(p) == N]
    
    if not valid_partitions:
        valid_partitions = [[13]]
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Try each partition
    for partition in valid_partitions:
        if time.time() > deadline - 5:
            break
        
        time_limit = min(deadline - 3, time.time() + 20)  # Max 20s per partition
        score, params = _optimize_partition(partition, time_limit)
        
        if score > best_overall_score and params is not None:
            best_overall_score = score
            global _current_partition
            _current_partition = partition
            best_overall_points = _make_points_from_params(params)
    
    # If optimization failed, fall back to regular 13-gon
    if best_overall_points is None or best_overall_score <= 0:
        angles = 2 * np.pi * np.arange(N) / N
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_overall_score = _score(best_overall_points)
    
    # Verify the result
    if best_overall_points is None or len(best_overall_points) != N:
        angles = 2 * np.pi * np.arange(N) / N
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_overall_points

# EVOLVE-BLOCK-END
