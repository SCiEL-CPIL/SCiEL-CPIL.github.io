"""Rewritten packing program with simulated annealing + rigid-body moves + soft-min surrogate."""
import numpy as np
import time

N = 200
DEADLINE = 115.0  # safety margin before 120s kill


def _score(pts):
    """Compute the maximum common radius for a set of centres in [-0.5,0.5]^2."""
    # Boundary distances
    dist_bound = np.min(np.minimum(0.5 - np.abs(pts), np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))), axis=1)
    # Actually compute per-circle min distance to boundary
    d_left = 0.5 + pts[:, 0]
    d_right = 0.5 - pts[:, 0]
    d_bottom = 0.5 + pts[:, 1]
    d_top = 0.5 - pts[:, 1]
    d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1)
    
    r = np.min(np.minimum(min_pair / 2.0, d_bound))
    return r


def _soft_min(x, beta):
    """Soft minimum via log-sum-exp trick: -log(sum(exp(-beta*x)))/beta."""
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-beta * (x - x_min)))) / beta


def _soft_score(pts, beta):
    """Smooth surrogate for the max-min objective."""
    d_left = 0.5 + pts[:, 0]
    d_right = 0.5 - pts[:, 0]
    d_bottom = 0.5 + pts[:, 1]
    d_top = 0.5 - pts[:, 1]
    d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1)
    
    # Combine: we want max of min(min_pair/2, d_bound)
    # Use soft-min over all constraints
    all_constraints = np.concatenate([min_pair / 2.0, d_bound])
    return _soft_min(all_constraints, beta)


def _grid_init():
    """Initialize with a square grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _hex_init():
    """Initialize with a hexagonal close-packed arrangement."""
    pts = []
    # Hexagonal packing: rows offset by half spacing
    # For 200 circles in unit square, try to fit as many rows as possible
    # Spacing s: horizontal = s, vertical = s*sqrt(3)/2
    # We want to fit N points. Try different spacings.
    
    best_pts = None
    best_score = -1
    
    for s in np.linspace(0.03, 0.07, 50):
        pts = []
        row = 0
        y = -0.5 + s * np.sqrt(3) / 2
        while y <= 0.5 - 1e-9:
            offset = 0.0 if row % 2 == 0 else s / 2
            x = -0.5 + s / 2 + offset
            while x <= 0.5 - 1e-9:
                pts.append([x, y])
                x += s
            row += 1
            y += s * np.sqrt(3) / 2
        
        pts = np.array(pts)
        if len(pts) >= N:
            # Take first N or try to center
            pts = pts[:N]
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
    
    if best_pts is not None:
        return best_pts
    return _grid_init()


def _find_clusters(pts, r_cluster):
    """Find clusters of circles within r_cluster of each other (connected components)."""
    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    for i in range(n):
        if visited[i]:
            continue
        # BFS
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            cur = queue.pop(0)
            for j in range(n):
                if not visited[j] and dists[cur, j] < r_cluster:
                    visited[j] = True
                    cluster.append(j)
                    queue.append(j)
        clusters.append(cluster)
    
    return clusters


def construct():
    start_time = time.time()
    
    # Initialize with hexagonal packing
    pts = _hex_init()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Simulated annealing with rigid-body moves
    n = len(pts)
    
    # Temperature schedule
    T_start = 0.02
    T_end = 0.0001
    # Multi-temperature: 4 levels
    temperatures = [T_start, T_start * 0.3, T_start * 0.1, T_end]
    
    # Beta schedule for soft-min surrogate
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    # Rigid-body cluster radius
    r_cluster = 0.08
    
    # Random number generator
    rng = np.random.default_rng(42)
    
    total_steps_target = 50000  # total steps to attempt
    
    step = 0
    for level, (T, beta) in enumerate(zip(temperatures, betas)):
        steps_this_level = total_steps_target // len(temperatures)
        
        for s in range(steps_this_level):
            # Check deadline
            if time.time() - start_time > DEADLINE:
                break
            
            # Choose move type: single-point (70%) or rigid-body (30%)
            if rng.random() < 0.3:
                # Rigid-body move: pick a cluster and move it
                clusters = _find_clusters(pts, r_cluster)
                if len(clusters) > 1:
                    # Pick a random cluster (not the largest to avoid trivial)
                    cluster_idx = rng.integers(0, len(clusters))
                    cluster = clusters[cluster_idx]
                    if len(cluster) > 1 and len(cluster) < n // 2:
                        # Random displacement
                        dx = rng.normal(0, T * 0.5, 1)
                        dy = rng.normal(0, T * 0.5, 1)
                        old_pts = pts.copy()
                        pts[cluster] += np.array([dx, dy])
                        
                        # Keep inside square
                        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
                        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
                        
                        new_score = _soft_score(pts, beta)
                        old_score = _soft_score(old_pts, beta)
                        
                        # Metropolis acceptance
                        delta = new_score - old_score
                        if delta > 0 or rng.random() < np.exp(delta / T):
                            pass  # accept
                        else:
                            pts = old_pts
            else:
                # Single-point move
                i = rng.integers(0, n)
                old_x, old_y = pts[i]
                pts[i, 0] += rng.normal(0, T)
                pts[i, 1] += rng.normal(0, T)
                
                # Keep inside square
                pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
                pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
                
                new_score = _soft_score(pts, beta)
                old_score = _soft_score(old_pts if 'old_pts' in dir() else pts.copy(), beta)
                
                # Recompute old score properly
                old_pts = pts.copy()
                pts[i, 0] = old_x
                pts[i, 1] = old_y
                old_score = _soft_score(pts, beta)
                pts[i, 0] += rng.normal(0, T)
                pts[i, 1] += rng.normal(0, T)
                pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
                pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
                new_score = _soft_score(pts, beta)
                
                delta = new_score - old_score
                if delta > 0 or rng.random() < np.exp(delta / T):
                    pass
                else:
                    pts[i, 0] = old_x
                    pts[i, 1] = old_y
            
            step += 1
        
        # After each temperature level, evaluate true score
        cur_score = _score(pts)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = pts.copy()
    
    # Final polish: gradient-like refinement
    if time.time() - start_time < DEADLINE - 10:
        # Do a final local optimization with very small steps
        T_polish = 0.00001
        for s in range(5000):
            if time.time() - start_time > DEADLINE - 5:
                break
            i = rng.integers(0, n)
            old_x, old_y = pts[i]
            pts[i, 0] += rng.normal(0, T_polish)
            pts[i, 1] += rng.normal(0, T_polish)
            pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
            pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
            
            cur_score = _score(pts)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = pts.copy()
            else:
                pts[i, 0] = old_x
                pts[i, 1] = old_y
    
    # Ensure best_pts is valid
    final_score = _score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        # Fallback to grid
        return _grid_init()
    
    return best_pts
