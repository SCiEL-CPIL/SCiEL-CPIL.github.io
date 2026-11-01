import numpy as np
import scipy.optimize as opt
import scipy.spatial as spat
import time
import itertools

N = 14

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Exact score: min triangle area / hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    # Compute all triangle areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _fast_score(points, k=20):
    """
    Fast approximate score: minimum area among the 'k' smallest triangles.
    This is a lower bound on the true score and much faster to compute 
    if we can prune the search space, but for N=14, 364 triangles is small.
    However, we can optimize the evaluation by precomputing indices.
    """
    n = len(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    # Precompute indices once outside if possible, but here we just vectorize
    # For N=14, C(14,3)=364. Vectorized computation is fast.
    # To make it "fast" for coordinate descent, we can skip the hull calculation 
    # if we assume the hull doesn't change much, or just compute it.
    # Actually, hull calculation is the bottleneck in CD.
    # We can approximate hull area as constant during fine-tuning if points stay in hull.
    # But for safety, let's just use the full score but optimize the triangle part.
    
    # The triangle part is already vectorized in _score.
    # The main cost is ConvexHull.
    # Let's try to reuse the hull if possible? No, points change.
    
    # Strategy for speed in CD:
    # 1. Compute triangle areas (fast)
    # 2. Find min triangle area
    # 3. Compute hull area (slow)
    # 4. Return ratio
    
    # If we assume the hull area is roughly constant during small perturbations,
    # we can skip hull calculation for candidate moves and only verify the best move.
    
    # Let's implement a version that returns min_area and hull_area separately?
    # Or just compute the full score.
    
    # For N=14, _score is ~0.5ms. 1000 steps/s is possible.
    # With 150s, we can do ~150k steps.
    # 14 points * 2 coords * 2 dirs = 56 moves per round.
    # 150k / 56 ~ 2600 rounds. That's plenty for CD.
    
    # So we can just use _score directly in CD if we optimize the Python loops.
    return _score(points)

def _generate_ring_config(radii, angles_per_ring, total_n):
    """
    Generate points based on ring radii and angular offsets.
    radii: list of radii for each ring
    angles_per_ring: list of number of points in each ring
    """
    points = []
    current_angle_offset = 0
    for i, (r, n_pts) in enumerate(zip(radii, angles_per_ring)):
        if n_pts == 0:
            continue
        # Angles for this ring
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + current_angle_offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points.extend(zip(x, y))
        
        # Stagger the next ring to avoid alignment
        # Offset by half the angular step of the current ring? Or fixed?
        # Let's offset by pi / n_pts
        current_angle_offset += np.pi / n_pts if n_pts > 0 else 0
        
    return np.array(points[:total_n])

def _objective_ring(params, ring_structure):
    """
    Objective for ring optimization.
    params: [r1, r2, ..., rK, theta1, theta2, ..., thetaK]
    """
    K = len(ring_structure)
    radii = params[:K]
    offsets = params[K:]
    
    # Ensure radii are positive and ordered
    radii = np.maximum(radii, 0.1)
    # Sort radii? No, we assign them to specific rings.
    # But we might want to enforce r1 < r2 < ... < rK
    # For simplicity, let's just use them as is, but penalize if not sorted?
    # Or just let the optimizer figure it out.
    
    points = []
    for i in range(K):
        n_pts = ring_structure[i]
        r = radii[i]
        theta_offset = offsets[i]
        
        if n_pts == 0:
            continue
            
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + theta_offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points.extend(zip(x, y))
        
    points = np.array(points)
    
    if len(points) < 3:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
        
    # Compute min triangle area
    n = len(points)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    # We want to maximize min_area / hull_area
    # So minimize - (min_area / hull_area)
    return -(min_area / hull_area)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Define possible ring structures for N=14
    # Examples: [14], [7, 7], [1, 13], [2, 12], [4, 10], [5, 9], [6, 8], [7, 7], [1, 6, 7], etc.
    # We'll pick a few promising ones.
    ring_structures = [
        [14],
        [7, 7],
        [1, 13],
        [2, 12],
        [4, 10],
        [5, 9],
        [6, 8],
        [1, 6, 7],
        [2, 5, 7],
        [3, 4, 7],
        [1, 5, 8],
        [2, 4, 8],
        [3, 3, 8],
        [4, 3, 7],
        [1, 4, 9],
        [2, 3, 9],
        [3, 2, 9],
        [4, 2, 8],
        [5, 2, 7],
        [6, 2, 6],
        [7, 2, 5],
        [8, 2, 4],
        [9, 2, 3],
        [10, 2, 2],
        [11, 2, 1],
        [12, 1, 1],
        [1, 1, 12],
        [2, 1, 11],
        [3, 1, 10],
        [4, 1, 9],
        [5, 1, 8],
        [6, 1, 7],
        [7, 1, 6],
        [8, 1, 5],
        [9, 1, 4],
        [10, 1, 3],
        [11, 1, 2],
        [12, 1, 1],
    ]
    
    # Remove duplicates and invalid ones (sum must be 14)
    unique_structures = []
    seen = set()
    for s in ring_structures:
        if sum(s) == N and len(s) > 0:
            # Sort to make unique (order of rings matters for radii, but structure is just counts)
            # Actually, order of rings matters because we assign radii r1, r2...
            # But [7,7] is the same as [7,7]. [1,13] is different from [13,1] in terms of which ring is inner.
            # Let's just keep them as is.
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique_structures.append(s)
                
    # Initialize best with a regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Optimize each ring structure
    for structure in unique_structures:
        if time.time() > deadline - 20:
            break
            
        K = len(structure)
        
        # Initial guess for radii: evenly spaced
        # Inner rings smaller, outer larger
        # For K=1, r=1
        # For K=2, r1=0.5, r2=1.0
        # For K=3, r1=0.3, r2=0.7, r3=1.0
        radii_init = np.linspace(0.2, 1.0, K)
        if K == 1:
            radii_init = np.array([1.0])
            
        # Initial guess for offsets: 0
        offsets_init = np.zeros(K)
        
        x0 = np.concatenate([radii_init, offsets_init])
        
        # Define bounds
        # Radii: [0.01, 1.5] (allow slightly >1 to expand hull if needed? No, normalize later)
        # Actually, we should normalize the points to unit area or just let the score handle it.
        # The score is min_area / hull_area. Scaling points by c scales min_area by c^2 and hull_area by c^2.
        # So the score is scale-invariant. We can fix the outermost radius to 1.
        # But if we have multiple rings, we can fix the largest radius to 1.
        # Let's just let the optimizer vary them, and the score will be invariant.
        
        bounds = []
        for i in range(K):
            bounds.append((0.01, 1.5)) # Radii
        for i in range(K):
            bounds.append((0, 2*np.pi)) # Offsets
            
        try:
            # Use L-BFGS-B
            result = opt.minimize(
                _objective_ring,
                x0,
                args=(structure,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < _objective_ring(x0, structure):
                opt_params = result.x
                opt_radii = opt_params[:K]
                opt_offsets = opt_params[K:]
                
                # Generate points
                points = []
                for i in range(K):
                    n_pts = structure[i]
                    r = opt_radii[i]
                    theta_offset = opt_offsets[i]
                    
                    if n_pts == 0:
                        continue
                        
                    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + theta_offset
                    x = r * np.cos(angles)
                    y = r * np.sin(angles)
                    points.extend(zip(x, y))
                    
                opt_points = np.array(points)
                
                if len(opt_points) == N:
                    opt_score = _score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
        except Exception:
            pass
            
    # Final polish with coordinate descent
    if time.time() < deadline - 10:
        current_points = best_points.copy()
        current_score = _score(current_points)
        
        # Adaptive step size
        step_size = 0.05
        no_improve_count = 0
        
        while time.time() < deadline - 2:
            improved = False
            for i in range(N):
                for coord in range(2):
                    for direction in [-1, 1]:
                        new_points = current_points.copy()
                        new_points[i, coord] += direction * step_size
                        
                        # Quick check: if point moves too far, skip?
                        # Just compute score
                        new_score = _score(new_points)
                        
                        if new_score > current_score:
                            current_points = new_points
                            current_score = new_score
                            improved = True
                            break # Move to next point/coord? Or continue?
                            # Breaking direction loop, but continuing coord loop?
                            # Let's restart the direction loop for this coord?
                            # No, just break out of direction loop.
                            # We should probably try the other direction too?
                            # If we improved, we can try to improve further in same direction?
                            # Simple CD: take the best move.
                            # Let's find the best move among all directions for this coord.
                            
            # Refine: For each point and coord, find the best step
            for i in range(N):
                for coord in range(2):
                    best_dir = 0
                    best_move_score = current_score
                    for direction in [-1, 1]:
                        new_points = current_points.copy()
                        new_points[i, coord] += direction * step_size
                        new_score = _score(new_points)
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_dir = direction
                            
                    if best_dir != 0:
                        current_points[i, coord] += best_dir * step_size
                        current_score = best_move_score
                        improved = True
                        
            if improved:
                no_improve_count = 0
                step_size *= 0.95
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                no_improve_count += 1
                step_size *= 0.5
                if step_size < 1e-5:
                    break
                    
    return best_points
