import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# Precompute triangle indices for vectorized area calculation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def generate_ring_points(counts, radii, angles):
    """
    Generate points from ring parameters.
    counts: list of integers, sum to N
    radii: list of floats, len = len(counts)
    angles: list of floats, len = len(counts)
    """
    points = []
    for i, count in enumerate(counts):
        if count == 0:
            continue
        # Regular polygon with 'count' vertices
        # Angles are offset by angles[i]
        base_angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        current_angles = base_angles + angles[i]
        r = radii[i]
        x = r * np.cos(current_angles)
        y = r * np.sin(current_angles)
        points.append(np.column_stack((x, y)))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    # Ensure exactly N points
    if len(result) < N:
        # This shouldn't happen if counts sum to N, but safety check
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
        
    return result

def optimize_partition(counts, time_budget):
    """
    Optimize radii and angles for a specific partition of N into rings.
    """
    k = len(counts)
    # Parameters: [r1, ..., rk, a1, ..., ak]
    # Radii should be positive and ordered r1 > r2 > ... > rk for non-crossing rings usually,
    # but differential evolution can handle unsorted if we sort later or just let it find the best.
    # However, for a "ring" structure, it's best to constrain r_i > r_{i+1}.
    # To simplify, we will optimize unsorted radii and angles, then sort points by radius?
    # No, the topology of the rings matters. Let's just optimize free radii and angles.
    
    bounds = []
    # Radii bounds: 0 to 1.5 (unit area normalization happens in score, but we can scale)
    # Actually, the score is scale invariant (ratio of areas). So we can fix one radius or just let it vary.
    # Let's let all radii vary in [0.1, 1.5].
    for _ in range(k):
        bounds.append((0.1, 1.5))
    # Angles bounds: [0, 2*pi]
    for _ in range(k):
        bounds.append((0, 2 * np.pi))
        
    def ring_objective(params):
        radii = params[:k]
        angles = params[k:]
        pts = generate_ring_points(counts, radii, angles)
        return -compute_score(pts)
        
    # Use Differential Evolution
    # Popsize should be reasonable. Maxiter limited by time.
    start_time = time.time()
    
    try:
        res = differential_evolution(
            ring_objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-5,
            popsize=8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,
            init='latinhypercube'
        )
        
        best_params = res.x
        best_pts = generate_ring_points(counts, best_params[:k], best_params[k:])
        return best_pts
        
    except Exception as e:
        # Fallback to a simple heuristic configuration for this partition
        # Place rings at geometrically spaced radii
        radii = np.linspace(1.0, 0.2, k)
        angles = np.zeros(k)
        return generate_ring_points(counts, radii, angles)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_points = None
    best_score = -float('inf')
    
    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Generate candidate partitions of 14
    # We want to try partitions with small number of rings (2, 3, 4)
    # Partitions of 14 into k parts
    candidate_partitions = []
    
    # k=2
    candidate_partitions.append([7, 7])
    candidate_partitions.append([8, 6])
    candidate_partitions.append([9, 5])
    
    # k=3
    candidate_partitions.append([6, 5, 3])
    candidate_partitions.append([5, 5, 4])
    candidate_partitions.append([7, 4, 3])
    candidate_partitions.append([8, 3, 3])
    
    # k=4
    candidate_partitions.append([5, 4, 3, 2])
    candidate_partitions.append([4, 4, 3, 3])
    candidate_partitions.append([6, 3, 3, 2])
    
    # Shuffle to ensure diversity if time runs out? No, prioritize likely good ones.
    # [7,7] and [5,5,4] are strong candidates.
    
    # Allocate time budget among partitions
    num_partitions = len(candidate_partitions)
    time_per_partition = (deadline - start_time - 10) / num_partitions # 10s for final polish
    
    for counts in candidate_partitions:
        if time.time() > deadline - 10:
            break
            
        # Ensure we have enough time for this partition
        if time.time() + 5 > deadline:
            break
            
        # Optimize this partition
        pts = optimize_partition(counts, time_per_partition)
        try_update(pts)
        
        # If we found a very good score, we might want to spend more time refining it later.
        # For now, just move to next partition.

    # Phase 2: Local refinement of the best candidate found
    if best_points is not None and time.time() < deadline - 5:
        # Try Nelder-Mead on the full 28D space
        # This allows breaking the ring symmetry if it helps
        x0 = best_points.flatten()
        
        try:
            res = minimize(
                neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False}
            )
            
            if res.fun < -best_score:
                pts = res.x.reshape((N, 2))
                try_update(pts)
                
        except Exception:
            pass

    # Fallback
    if best_points is None:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

# Test to ensure it runs
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {compute_score(pts)}")
    print(f"Points:\n{pts}")
