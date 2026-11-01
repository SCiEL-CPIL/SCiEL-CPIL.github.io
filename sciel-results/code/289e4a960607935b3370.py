"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_objective(params, alpha=50.0):
    """Smooth surrogate: minimize -soft_min of all constraints.
    The soft min approximates min of pairwise distances/2 and boundary distances.
    We maximize this, so minimize its negative.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Boundary constraints: 0.5 - |c_ik|
    boundary_dists = 0.5 - np.abs(c)
    flat_boundary = boundary_dists.flatten()
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean') / 2.0
    
    # Combine all constraints into one array
    all_vals = np.concatenate([flat_boundary, dists])
    
    # Soft min via log-sum-exp trick: soft_min(x) = -1/alpha * log(sum(exp(-alpha*x)))
    # To avoid overflow, subtract min
    min_val = np.min(all_vals)
    if min_val <= 0:
        return 1.0 - min_val  # penalty if any constraint violated
    
    scaled = -alpha * (all_vals - min_val)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = min_val - log_sum / alpha
    
    return -soft_min


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.085, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_grid():
    """Initialize with a uniform grid pattern."""
    best_c = None
    best_r = 0
    
    for cols in range(13, 17):
        rows = int(np.ceil(N / cols))
        if rows * cols < N:
            continue
        
        x_ticks = (np.arange(cols) + 0.5) / cols - 0.5
        y_ticks = (np.arange(rows) + 0.5) / rows - 0.5
        
        grid = np.array([(x, y) for y in y_ticks for x in x_ticks])
        if len(grid) >= N:
            c = grid[:N].copy()
            r = _compute_radius(c)
            if best_c is None or r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=25.0, alpha=50.0):
    """Local optimization using L-BFGS-B with smooth surrogate objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Pass 1: medium alpha for smoothness
    try:
        result = minimize(
            _soft_min_objective,
            c0.flatten(),
            args=(alpha,),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    except Exception:
        pass
    
    # Pass 2: higher alpha for sharper approximation
    elapsed = time.time() - start_time
    if elapsed < max_time - 5:
        try:
            alpha2 = alpha * 3
            result2 = minimize(
                _soft_min_objective,
                best_c.flatten(),
                args=(alpha2,),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with two passes on exact objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _objective(params):
        c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r = _compute_radius(c)
        if r <= 0:
            return 1.0
        return -r
    
    # First pass: coarse optimization
    try:
        result = minimize(
            _objective,
            c0.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    except Exception:
        pass
    
    # Second pass: fine optimization if time permits
    elapsed = time.time() - start_time
    if elapsed < max_time - 5:
        try:
            result2 = minimize(
                _objective,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _force_directed_refine(c0, max_time=20.0):
    """Force-directed relaxation: push overlapping circles apart, pull toward center if near boundary."""
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    # Use the current radius as the target separation
    target_sep = 2.0 * r
    
    for iteration in range(500):
        elapsed = time.time() - start_time
        if elapsed > max_time - 2:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        dists = pdist(c, metric='euclidean')
        # Reconstruct full distance matrix
        n = c.shape[0]
        dist_matrix = np.zeros((n, n))
        idx = np.triu_indices(n, k=1)
        dist_matrix[idx] = dists
        dist_matrix = dist_matrix + dist_matrix.T
        
        # For pairs closer than target_sep, apply repulsive force
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if d < target_sep and d > 1e-10:
                    direction = (c[i] - c[j]) / d
                    force_mag = 0.5 * (target_sep - d)
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
        
        # Boundary forces: if circle is too close to boundary, push inward
        for i in range(n):
            for k in range(2):
                if c[i, k] > 0.5 - r:
                    forces[i, k] -= 0.5 * (c[i, k] - (0.5 - r))
                elif c[i, k] < -0.5 + r:
                    forces[i, k] += 0.5 * ((-0.5 + r) - c[i, k])
        
        # Apply forces with damping
        c += 0.1 * forces
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        r_new = _compute_radius(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
            # Update target separation
            target_sep = 2.0 * best_r
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_c = None
    best_r = 0.0
    
    # 1. Try hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_r:
        best_r = r_hex
        best_c = c_hex.copy()
    
    # 2. Try grid initialization
    c_grid = _initialize_grid()
    r_grid = _compute_radius(c_grid)
    if r_grid > best_r:
        best_r = r_grid
        best_c = c_grid.copy()
    
    # 3. Optimize the best initialization using smooth surrogate
    elapsed = time.time() - start_time
    if elapsed < deadline - 50:
        c_opt, r_opt = _local_optimize_smooth(best_c, max_time=min(30.0, deadline - time.time() - 45))
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # 4. Polish with exact objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        c_polish, r_polish = _local_optimize(best_c, max_time=min(20.0, deadline - time.time() - 25))
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
    
    # 5. Force-directed refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        c_fd, r_fd = _force_directed_refine(best_c, max_time=min(15.0, deadline - time.time() - 12))
        if r_fd > best_r:
            best_r = r_fd
            best_c = c_fd.copy()
    
    # 6. Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 8:
        c_final, r_final = _local_optimize(best_c, max_time=min(10.0, deadline - time.time() - 6))
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # Final safety check
    if best_c is None:
        best_c = c_hex
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
