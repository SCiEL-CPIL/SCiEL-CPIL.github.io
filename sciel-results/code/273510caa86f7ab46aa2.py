import numpy as np
from itertools import combinations
import time

def _compute_areas(points):
    """Compute all doubled absolute areas for triples of points using NumPy."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    # For each triple (i,j,k), doubled area = |det(pts[j]-pts[i], pts[k]-pts[i])|
    # Use broadcasting: for all i, j, k
    # points[i], points[j], points[k]
    # We'll compute for all i < j < k to save memory
    # But for vectorization, let's do it differently
    # For each pair (i,j), compute area with all k
    # Actually, let's just use the standard approach
    idx = np.arange(n)
    # points[i] - points[j] for all i,j
    # This gets memory heavy for n=30, but 30^3 = 27000 is fine
    pi = pts[:, np.newaxis, :]  # (n, 1, 2)
    pj = pts[np.newaxis, :, :]  # (1, n, 2)
    # (pj - pi) shape (n, n, 2)
    v1 = pj - pi  # (n, n, 2): v1[i,j] = pts[j] - pts[i]
    # Now for each (i,j), compute cross product magnitude with pts[k] - pts[i]
    # v2[i,j,k] = pts[k] - pts[i]
    pk = pts[np.newaxis, np.newaxis, :]  # (1, 1, n, 2)
    # v2[i,j,k] = pts[k] - pts[i]
    # Let's compute differently
    # For triple (i,j,k): area2 = |det(pts[j]-pts[i], pts[k]-pts[i])|
    # = |(pts[j][0]-pts[i][0])*(pts[k][1]-pts[i][1]) - (pts[j][1]-pts[i][1])*(pts[k][0]-pts[i][0])|
    
    # Let's compute for all i < j < k
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                areas.append(abs(cross))
    return np.array(areas, dtype=np.float64)

def _compute_areas_fast(points):
    """Vectorized computation of all doubled absolute areas."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    
    # Use index arrays for i < j < k
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    # Filter to i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_sel = i_idx[mask]
    j_sel = j_idx[mask]
    k_sel = k_idx[mask]
    
    pi = pts[i_sel]  # (m, 2)
    pj = pts[j_sel]  # (m, 2)
    pk = pts[k_sel]  # (m, 2)
    
    # cross product
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(cross)
    return areas

def _max_multiplicity(areas):
    """Find the maximum multiplicity of any positive doubled area."""
    positive_areas = areas[areas > 0.5]  # Use 0.5 threshold to avoid floating point issues
    if len(positive_areas) == 0:
        return 0
    # Round to nearest integer for counting (doubled areas should be integers for integer coords)
    rounded = np.round(positive_areas).astype(np.int64)
    # Count occurrences
    unique, counts = np.unique(rounded, return_counts=True)
    return int(np.max(counts))

def _score(points):
    """Compute the score for a set of points."""
    areas = _compute_areas_fast(points)
    return _max_multiplicity(areas)

def _softmax_score(areas, temperature=1.0):
    """Smoothed approximation of max multiplicity using log-sum-exp."""
    positive_areas = areas[areas > 0.5]
    if len(positive_areas) == 0:
        return 0.0
    rounded = np.round(positive_areas).astype(np.int64)
    unique, counts = np.unique(rounded, return_counts=True)
    # Softmax-like: sum(exp(counts / temperature))
    # This approximates max when temperature is small
    scaled = counts / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(log_sum_exp)

def _local_refine(points, max_iters=200, time_limit=5.0):
    """Local search refinement: try moving individual points to nearby integer positions."""
    start_time = time.time()
    best_score = _score(points)
    best_points = list(points)
    n = len(points)
    
    # Precompute all valid positions
    valid_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            current_pos = points[i]
            # Try nearby positions (within radius 3)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = current_pos[0] + dx
                    new_y = current_pos[1] + dy
                    if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                        continue
                    new_pos = (new_x, new_y)
                    # Check if this position is already used
                    if new_pos in points:
                        continue
                    
                    new_points = list(points)
                    new_points[i] = new_pos
                    new_score = _score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
                        points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_score

def _random_init():
    """Generate a random set of 30 distinct integer points in [-30,30]^2."""
    np.random.seed(42)  # Fixed seed for reproducibility
    points = []
    used = set()
    while len(points) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        if (x, y) not in used:
            points.append((x, y))
            used.add((x, y))
    return points

def _structured_init():
    """Generate a structured initial set of points."""
    # Grid-like structure with some perturbation
    points = []
    used = set()
    # Start with a 6x5 grid centered
    for i in range(6):
        for j in range(5):
            x = -15 + i * 5
            y = -10 + j * 5
            points.append((x, y))
            used.add((x, y))
    return points

def construct():
    """Construct 30 distinct integer coordinate pairs in [-30,30]^2 to maximize
    the largest multiplicity of a single positive doubled area."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Try multiple initialization strategies
    best_points = None
    best_score = 0
    
    # Strategy 1: Structured grid
    points1 = _structured_init()
    score1 = _score(points1)
    if best_points is None or score1 > best_score:
        best_points = list(points1)
        best_score = score1
    
    # Strategy 2: Random initialization
    points2 = _random_init()
    score2 = _score(points2)
    if score2 > best_score:
        best_points = list(points2)
        best_score = score2
    
    # Local refinement on the best initial solution
    if time.time() < deadline - 10.0:
        refined_points, refined_score = _local_refine(best_points, max_iters=100, time_limit=15.0)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    # Try a few more random restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 15.0:
            break
        
        np.random.seed(42 + restart * 100)
        points_r = _random_init()
        score_r = _score(points_r)
        
        if score_r > best_score:
            best_points = list(points_r)
            best_score = score_r
        
        # Local refine if promising
        if time.time() < deadline - 10.0 and score_r >= best_score - 2:
            refined_r, refined_score_r = _local_refine(points_r, max_iters=50, time_limit=8.0)
            if refined_score_r > best_score:
                best_points = refined_r
                best_score = refined_score_r
    
    # Final local refinement with more iterations
    if time.time() < deadline - 5.0:
        final_points, final_score = _local_refine(best_points, max_iters=50, time_limit=5.0)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    return best_points

# Keep the original function signature
def construct():
    """Construct 30 distinct integer coordinate pairs in [-30,30]^2 to maximize
    the largest multiplicity of a single positive doubled area."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Try multiple initialization strategies
    best_points = None
    best_score = 0
    
    # Strategy 1: Structured grid
    points1 = _structured_init()
    score1 = _score(points1)
    if best_points is None or score1 > best_score:
        best_points = list(points1)
        best_score = score1
    
    # Strategy 2: Random initialization
    points2 = _random_init()
    score2 = _score(points2)
    if score2 > best_score:
        best_points = list(points2)
        best_score = score2
    
    # Local refinement on the best initial solution
    if time.time() < deadline - 10.0:
        refined_points, refined_score = _local_refine(best_points, max_iters=100, time_limit=15.0)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    # Try a few more random restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 15.0:
            break
        
        np.random.seed(42 + restart * 100)
        points_r = _random_init()
        score_r = _score(points_r)
        
        if score_r > best_score:
            best_points = list(points_r)
            best_score = score_r
        
        # Local refine if promising
        if time.time() < deadline - 10.0 and score_r >= best_score - 2:
            refined_r, refined_score_r = _local_refine(points_r, max_iters=50, time_limit=8.0)
            if refined_score_r > best_score:
                best_points = refined_r
                best_score = refined_score_r
    
    # Final local refinement with more iterations
    if time.time() < deadline - 5.0:
        final_points, final_score = _local_refine(best_points, max_iters=50, time_limit=5.0)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    return best_points
