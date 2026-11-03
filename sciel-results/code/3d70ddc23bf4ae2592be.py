"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist

N = 200


def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < 0:
        return 0.0
    # Min pairwise distance
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)


def _hex_grid_init(rows, cols, jitter=0.0):
    """Generate a hexagonal grid initialization with optional jitter."""
    points = []
    # Calculate spacing to fit in the square
    # For a hex grid, the horizontal spacing is s, vertical is s*sqrt(3)/2
    # Width approx: cols * s
    # Height approx: rows * s * sqrt(3)/2
    s_x = 1.0 / cols if cols > 0 else 1.0
    s_y = 1.0 / (max(1, (rows - 1) * np.sqrt(3) / 2 + 1)) if rows > 1 else 1.0
    s = min(s_x, s_y)
    
    count = 0
    for j in range(rows):
        offset = (j % 2) * s / 2
        n_in_row = cols
        if count + n_in_row > N:
            n_in_row = N - count
        for i in range(n_in_row):
            x = -0.5 + s / 2 + i * s + offset
            y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
            # Add jitter
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            # Clip to bounds (slightly inside to avoid boundary issues initially)
            x = np.clip(x, -0.49, 0.49)
            y = np.clip(y, -0.49, 0.49)
            points.append([x, y])
            count += 1
            if count >= N:
                break
        if count >= N:
            break
    
    if len(points) < N:
        return None
    return np.array(points[:N])


def _square_grid_init(m, jitter=0.0):
    """Generate a square grid initialization."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) < N:
        return None
    points = grid[:N].copy()
    if jitter > 0:
        points += np.random.uniform(-jitter, jitter, points.shape)
        points = np.clip(points, -0.49, 0.49)
    return points


def _local_refine(points, time_budget=10.0, step_init=0.01):
    """
    Local refinement using targeted bottleneck elimination.
    Moves only the points involved in the current minimum distance constraint.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    step = step_init
    no_improve_count = 0
    
    def _get_bottleneck(pts):
        """Returns (type, value, indices) where type is 'pair' or 'boundary'."""
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        min_boundary_idx = np.argmin(dist_boundary)
        
        d = pdist(pts)
        min_pair = np.min(d)
        min_pair_idx = np.argmin(d)
        
        # Convert condensed index to (i, j)
        n = N
        k = min_pair_idx
        for i in range(n):
            max_j = n - i - 1
            if k < max_j:
                j = i + 1 + k
                break
            k -= max_j
        pair = (i, j)
        
        if min_pair / 2.0 < min_boundary:
            return 'pair', min_pair / 2.0, pair
        else:
            return 'boundary', min_boundary, min_boundary_idx
    
    while time.time() - start_time < time_budget:
        improved = False
        btype, bval, bidx = _get_bottleneck(points)
        
        if btype == 'pair':
            i, j = bidx
            # Try to move i and j apart
            # Direction from j to i
            diff = points[i] - points[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                dir_vec = diff / norm
            else:
                dir_vec = np.array([1.0, 0.0])
            
            # Try moving i in dir_vec direction and j in -dir_vec direction
            candidates = []
            # Move both
            new_i = points[i] + dir_vec * step
            new_j = points[j] - dir_vec * step
            if np.all(new_i >= -0.5) and np.all(new_i <= 0.5) and np.all(new_j >= -0.5) and np.all(new_j <= 0.5):
                new_points = points.copy()
                new_points[i] = new_i
                new_points[j] = new_j
                candidates.append(new_points)
            
            # Move i only
            new_i = points[i] + dir_vec * step
            if np.all(new_i >= -0.5) and np.all(new_i <= 0.5):
                new_points = points.copy()
                new_points[i] = new_i
                candidates.append(new_points)
            
            # Move j only
            new_j = points[j] - dir_vec * step
            if np.all(new_j >= -0.5) and np.all(new_j <= 0.5):
                new_points = points.copy()
                new_points[j] = new_j
                candidates.append(new_points)
            
            # Also try perpendicular directions for one of them to escape local trap
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            for sign in [1, -1]:
                new_i = points[i] + dir_vec * step * 0.5 + perp_vec * step * sign * 0.5
                if np.all(new_i >= -0.5) and np.all(new_i <= 0.5):
                    new_points = points.copy()
                    new_points[i] = new_i
                    candidates.append(new_points)
                
                new_j = points[j] - dir_vec * step * 0.5 + perp_vec * step * sign * 0.5
                if np.all(new_j >= -0.5) and np.all(new_j <= 0.5):
                    new_points = points.copy()
                    new_points[j] = new_j
                    candidates.append(new_points)
            
            for cand in candidates:
                new_r = _compute_radius(cand)
                if new_r > current_r + 1e-12:
                    current_r = new_r
                    points = cand
                    improved = True
                    break
            
        elif btype == 'boundary':
            i = bidx
            # Move i towards center
            # Direction is -points[i] normalized
            norm = np.linalg.norm(points[i])
            if norm > 1e-12:
                dir_vec = -points[i] / norm
            else:
                dir_vec = np.array([1.0, 0.0])
            
            new_i = points[i] + dir_vec * step
            if np.all(new_i >= -0.5) and np.all(new_i <= 0.5):
                new_points = points.copy()
                new_points[i] = new_i
                new_r = _compute_radius(new_points)
                if new_r > current_r + 1e-12:
                    current_r = new_r
                    points = new_points
                    improved = True
            
            # If boundary move didn't work, try small random jitter to escape
            if not improved:
                for _ in range(5):
                    new_i = points[i] + np.random.normal(0, step, 2)
                    if np.all(new_i >= -0.5) and np.all(new_i <= 0.5):
                        new_points = points.copy()
                        new_points[i] = new_i
                        new_r = _compute_radius(new_points)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            points = new_points
                            improved = True
                            break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 5:
                step *= 0.5
                if step < 1e-7:
                    break
                no_improve_count = 0
        else:
            # Keep step size or slightly increase if we are far from optimal
            pass
    
    return points, current_r


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_points = None
    best_r = 0.0
    
    # Initialize with multiple strategies
    inits = []
    
    # 1. Hex grids with different aspect ratios
    for rows in range(12, 22):
        cols = int(np.ceil(N / rows))
        if cols < 5:
            continue
        pts = _hex_grid_init(rows, cols, jitter=0.0)
        if pts is not None:
            inits.append(pts)
    
    # 2. Square grids
    for m in range(13, 16):
        pts = _square_grid_init(m, jitter=0.0)
        if pts is not None:
            inits.append(pts)
    
    # 3. Jittered hex grids (random restarts)
    np.random.seed(42)
    for _ in range(5):
        rows = np.random.randint(13, 20)
        cols = int(np.ceil(N / rows))
        pts = _hex_grid_init(rows, cols, jitter=0.01)
        if pts is not None:
            inits.append(pts)
    
    # Evaluate and refine each initialization
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
        
        r = _compute_radius(init_pts)
        if r > best_r:
            best_r = r
            best_points = init_pts.copy()
        
        # Refine the best candidate found so far, or this one if it's better
        # To save time, we only refine if it has potential to beat best_r
        # Or we refine the current best periodically.
        # Let's refine this specific initialization if it's promising or if we have time.
        # Strategy: Refine the current global best every few iterations?
        # Better: Refine this init, and if it beats global best, update.
        
        remaining = deadline - time.time() - 5
        if remaining > 5:
            # Allocate time budget for refinement based on remaining time
            # We want to try multiple inits, so budget per init is limited
            budget = min(remaining / 2.0, 15.0)  # Max 15s per refinement, at least half remaining? No, that's too much.
            # Let's use a fixed budget per refinement to ensure we try several
            budget = 10.0
            if remaining < budget:
                budget = remaining - 2.0
            
            if budget > 2.0:
                refined_pts, refined_r = _local_refine(init_pts, time_budget=budget)
                if refined_r > best_r:
                    best_r = refined_r
                    best_points = refined_pts.copy()
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            best_points, best_r = _local_refine(best_points, time_budget=remaining, step_init=0.005)
    
    if best_points is None:
        # Fallback
        best_points = _square_grid_init(14, jitter=0.0)
        if best_points is None:
            best_points = np.zeros((N, 2))
    
    return best_points
