"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import LinearConstraint


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    # Combine all distance components
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
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
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _rigid_body_anneal(c0, max_time=30.0, seed=42):
    """Simulated annealing with rigid-body cluster moves to escape local minima."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    current_c = c0.copy()
    current_r = best_r
    
    # Build neighbor graph for cluster identification
    # Points within 1.5x current radius are considered neighbors
    def _build_clusters(c, r):
        """Identify clusters of nearby points using union-find."""
        n = len(c)
        threshold = max(r * 1.8, 0.04)
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Find neighbors using grid-based spatial hashing
        cell_size = threshold
        grid = {}
        for i in range(n):
            cx = int((c[i, 0] + 0.5) / cell_size)
            cy = int((c[i, 1] + 0.5) / cell_size)
            grid.setdefault((cx, cy), []).append(i)
        
        for i in range(n):
            cx = int((c[i, 0] + 0.5) / cell_size)
            cy = int((c[i, 1] + 0.5) / cell_size)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for j in grid.get((cx + dx, cy + dy), []):
                        if j > i:
                            dist = np.linalg.norm(c[i] - c[j])
                            if dist < threshold:
                                union(i, j)
        
        # Group by cluster
        clusters = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)
        
        return list(clusters.values())
    
    # Initial clusters
    clusters = _build_clusters(current_c, current_r)
    
    # Annealing parameters
    T_start = 0.002
    T_end = 0.00001
    n_iterations = 5000
    move_scale = 0.008
    
    for it in range(n_iterations):
        if time.time() > start_time + max_time - 0.5:
            break
        
        # Temperature schedule (geometric cooling)
        progress = it / n_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Choose move type: single point (70%) or cluster (30%)
        if rng.random() < 0.7:
            # Single point move
            idx = rng.randint(N)
            idxs = [idx]
        else:
            # Cluster move: pick a random cluster
            if not clusters:
                idxs = [rng.randint(N)]
            else:
                cluster = clusters[rng.randint(len(clusters))]
                # Limit cluster size to avoid huge moves
                if len(cluster) > 8:
                    idxs = list(rng.choice(cluster, size=8, replace=False))
                else:
                    idxs = cluster
        
        # Generate random displacement
        disp = rng.normal(0, move_scale, size=(len(idxs), 2))
        new_c = current_c.copy()
        new_c[idxs] += disp
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Compute new radius
        new_r = _compute_radius(new_c)
        
        # Accept/reject
        delta_r = new_r - current_r
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_c = new_c
            current_r = new_r
            
            # Rebuild clusters periodically
            if it % 200 == 0:
                clusters = _build_clusters(current_c, current_r)
            
            # Track best
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
    
    return best_c, best_r


def _high_precision_polish(c0, max_time=15.0):
    """High-precision polish: use very high beta and tight tolerances, then rigid-body anneal."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Very high beta values for near-exact min
    betas = [1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Rigid-body annealing to escape local minima
    if time.time() < start_time + max_time - 1:
        remaining = max(2.0, start_time + max_time - time.time() - 1)
        c_anneal, r_anneal = _rigid_body_anneal(best_c, max_time=remaining, seed=123)
        if r_anneal > best_r:
            best_c = c_anneal
            best_r = r_anneal
    
    return best_c, best_r


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high with more intermediate steps
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 120, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # High-precision polish if time permits
    if time.time() < start_time + max_time - 2:
        c_polished, r_polished = _high_precision_polish(
            best_c, max_time=max(2.0, max_time - (time.time() - start_time) - 2)
        )
        if r_polished > best_r:
            best_c = c_polished
            best_r = r_polished
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Smooth optimization from hexagonal start (primary optimization)
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Rigid-body annealing to escape local minima
    if best_c is not None and time.time() < deadline - 40:
        remaining = min(35.0, deadline - time.time() - 40)
        c_anneal, r_anneal = _rigid_body_anneal(best_c, max_time=remaining, seed=42)
        if r_anneal > best_radius:
            best_radius = r_anneal
            best_c = c_anneal.copy()
    
    # Second round: smooth optimize from the annealed result
    if best_c is not None and time.time() < deadline - 25:
        remaining = min(20.0, deadline - time.time() - 25)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=remaining)
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # Another annealing pass from the new best
    if best_c is not None and time.time() < deadline - 15:
        remaining = min(12.0, deadline - time.time() - 15)
        c_anneal2, r_anneal2 = _rigid_body_anneal(best_c, max_time=remaining, seed=99)
        if r_anneal2 > best_radius:
            best_radius = r_anneal2
            best_c = c_anneal2.copy()
    
    # Final high-precision polish on the best solution found
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_final, r_final = _high_precision_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
