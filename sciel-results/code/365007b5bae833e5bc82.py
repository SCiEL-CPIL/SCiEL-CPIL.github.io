"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize

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


def _hex_grid_initialization(num_points, jitter=0.0):
    """
    Generate a hexagonal grid initialization for num_points.
    If jitter > 0, add random perturbation to break symmetry.
    """
    # Find good grid dimensions
    best_points = None
    best_r = 0.0
    
    for rows in range(12, 25):
        cols = num_points // rows
        rem = num_points % rows
        
        if cols < 1:
            continue
            
        # Hex grid spacing
        # We want to fit cols points across and rows points down
        # With hex offset, the effective width is about cols * s
        # The height is about rows * s * sqrt(3) / 2
        
        # Try different spacings
        for s_factor in np.linspace(0.8, 1.2, 10):
            s_x = 1.0 / cols
            s_y = 1.0 / (rows * np.sqrt(3) / 2)
            s = min(s_x, s_y) * s_factor
            
            points = []
            count = 0
            for j in range(rows):
                offset = (j % 2) * s / 2
                n_in_row = cols + (1 if j < rem else 0)
                for i in range(n_in_row):
                    x = -0.5 + s / 2 + i * s + offset
                    y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        points.append([x, y])
                        count += 1
                        if count >= num_points:
                            break
                if count >= num_points:
                    break
            
            if len(points) == num_points:
                points = np.array(points)
                
                # Add jitter if requested
                if jitter > 0:
                    rng = np.random.RandomState(42)
                    points += rng.uniform(-jitter, jitter, size=points.shape)
                    # Project back to valid region
                    points = np.clip(points, -0.5, 0.5)
                
                r = _compute_radius(points)
                if r > best_r:
                    best_r = r
                    best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: square grid
    m = int(np.ceil(np.sqrt(num_points)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:num_points]


def _soft_min_objective(x, beta=100.0):
    """
    Smooth approximation of the negative packing radius.
    Uses soft-min to handle the min of pairwise distances and boundary distances.
    """
    points = x.reshape(-1, 2)
    
    # Boundary distances
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    
    # Pairwise distances
    d = pdist(points)
    
    # Combine all distances (pairwise + boundary)
    # We want to maximize the minimum, so minimize the negative
    # Soft-min: -log-sum-exp(-d/beta) / beta
    all_dists = np.concatenate([d, dist_boundary])
    
    # For numerical stability
    min_d = np.min(all_dists)
    exp_dists = np.exp(-(all_dists - min_d) / beta)
    soft_min = min_d - beta * np.log(np.sum(exp_dists))
    
    return -soft_min / 2.0  # Negative because we're minimizing


def _nelder_mead_refine(points, time_budget=30.0, beta_start=50.0, beta_end=200.0):
    """
    Refine using Nelder-Mead with annealing beta for soft-min objective.
    """
    start_time = time.time()
    x = points.reshape(-1).copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    # Multi-stage optimization with increasing beta
    stages = [(50.0, 0.3), (100.0, 0.4), (200.0, 0.3)]  # (beta, time_fraction)
    
    for beta, time_frac in stages:
        stage_budget = time_budget * time_frac
        stage_deadline = start_time + (time.time() - start_time) + stage_budget
        
        if time.time() > stage_deadline - 1:
            continue
        
        # Nelder-Mead options
        options = {
            'maxiter': 500,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        try:
            result = minimize(
                _soft_min_objective,
                x,
                args=(beta,),
                method='Nelder-Mead',
                options=options,
                callback=None
            )
            
            new_points = result.x.reshape(-1, 2)
            # Ensure valid
            if np.all(np.isfinite(new_points)) and np.all((new_points >= -0.5) & (new_points <= 0.5)):
                new_r = _compute_radius(new_points)
                if new_r > current_r:
                    current_r = new_r
                    x = result.x
                    points = new_points
        except Exception:
            pass
    
    return points, current_r


def _local_refine(points, time_budget=10.0):
    """
    Local refinement using coordinate ascent with adaptive step size.
    For each point, try to move it in a direction that increases the min distance.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    step = 0.005
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Try moving point i in 8 directions + diagonal
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                        new_r = _compute_radius(new_points)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            points = new_points
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Reduce step size
                step *= 0.5
                if step < 1e-6:
                    break
                no_improve_count = 0
        else:
            # Keep step size or slightly increase
            pass
    
    return points, current_r


def _local_refine(points, time_budget=10.0):
    """
    Local refinement using coordinate ascent with adaptive step size.
    For each point, try to move it in a direction that increases the min distance.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    # Determine the bottleneck: is it a pair or the boundary?
    def _get_bottleneck(pts):
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        min_boundary_idx = np.argmin(dist_boundary)
        
        if N >= 2:
            d = pdist(pts)
            min_pair = np.min(d)
            # Find which pair
            min_pair_idx = np.argmin(d)
            # Convert from condensed index to (i, j)
            n = N
            # Invert triangular index
            k = min_pair_idx
            # i < j, index = i*n - i*(i+1)/2 + (j - i - 1)
            # Solve for i, j
            for i in range(n):
                max_j = n - i - 1
                if k < max_j:
                    j = i + 1 + k
                    break
                k -= max_j
            pair = (i, j)
        else:
            min_pair = float('inf')
            pair = None
        
        if min_pair / 2.0 < min_boundary:
            return 'pair', min_pair / 2.0, pair
        else:
            return 'boundary', min_boundary, min_boundary_idx
    
    step = 0.005
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Try moving point i in 8 directions + diagonal
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                        new_r = _compute_radius(new_points)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            points = new_points
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Reduce step size
                step *= 0.5
                if step < 1e-6:
                    break
                no_improve_count = 0
        else:
            # Keep step size or slightly increase
            pass
    
    return points, current_r


def construct():
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_points = None
    best_r = 0.0
    
    # Strategy 1: Hex grid initialization
    hex_points = _hex_grid_initialization(N, jitter=0.0)
    if hex_points is not None:
        r = _compute_radius(hex_points)
        if r > best_r:
            best_r = r
            best_points = hex_points.copy()
    
    # Strategy 2: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    sq_points = grid[:N]
    r = _compute_radius(sq_points)
    if r > best_r:
        best_r = r
        best_points = sq_points.copy()
    
    if best_points is None:
        # Fallback
        best_points = np.zeros((N, 2))
        return best_points
    
    # Nelder-Mead refinement with soft-min objective
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 15
        if remaining > 10:
            best_points, best_r = _nelder_mead_refine(best_points, time_budget=remaining)
    
    # Final local refinement
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            best_points, best_r = _local_refine(best_points, time_budget=remaining)
    
    return best_points
