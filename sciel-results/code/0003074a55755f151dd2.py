import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _score(centers):
    """Calculate the radius (score) for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(np.min(dists_sq))
    
    dists_boundary = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_boundary = np.min(dists_boundary)
    
    return min(min_dist, min_boundary) / 2.0

def _hexagonal_lattice(N, scale=1.0, offset_x=0.0, offset_y=0.0):
    """
    Generate N points in a hexagonal lattice pattern.
    Returns array of shape (N, 2).
    """
    # Hexagonal lattice: rows offset by half spacing
    # For N points, use roughly sqrt(N) rows
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    cols_needed = int(np.ceil(np.sqrt(N * np.sqrt(3) / 2)))
    
    points = []
    for i in range(rows):
        row_offset = (i % 2) * 0.5
        for j in range(cols_needed):
            x = (j + row_offset) * scale
            y = i * scale * np.sqrt(3) / 2
            points.append([x, y])
        if len(points) >= N:
            break
    
    points = np.array(points[:N])
    
    # Center the lattice
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    points[:, 0] -= cx
    points[:, 1] -= cy
    
    # Scale to fit in [-0.5, 0.5]
    max_extent = np.max(np.abs(points))
    if max_extent > 0:
        points *= (0.45 / max_extent)
    
    points += np.array([offset_x, offset_y])
    points = np.clip(points, -0.5, 0.5)
    
    return points

def _soft_min_energy(centers, temperature=0.01):
    """
    Soft-min surrogate energy that approximates maximizing the minimum distance.
    Uses a softmax-like weighting to emphasize close pairs.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    dists_sq = dists_sq[mask]
    
    # Soft minimum: -log(sum(exp(-d^2 / T)))
    # This emphasizes small distances
    dists_sq = np.maximum(dists_sq, 1e-12)
    weights = np.exp(-dists_sq / (2 * temperature))
    energy = -np.log(np.sum(weights) + 1e-12)
    
    # Boundary penalty
    boundary_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    boundary_penalty = np.sum(np.exp(-boundary_dists / temperature))
    
    return energy + boundary_penalty

def _local_polish(centers, free_indices, time_budget=5.0):
    """
    Polish only the free indices using L-BFGS-B.
    Returns improved centers.
    """
    start = time.time()
    n_free = len(free_indices)
    
    def objective(x_free):
        c = centers.copy()
        c[free_indices] = x_free
        c = np.clip(c, -0.5, 0.5)
        return _soft_min_energy(c, temperature=0.005)
    
    x0 = centers[free_indices].flatten()
    bounds = [(-0.5, 0.5)] * (2 * n_free)
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if res.success or res.fun < objective(x0):
            centers[free_indices] = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centers

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    
    # Best solution tracking
    best_centers = None
    best_score = 0.0
    
    # Try multiple hexagonal configurations with different scales and offsets
    configs = []
    for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
        for ox in [-0.05, 0.0, 0.05]:
            for oy in [-0.05, 0.0, 0.05]:
                configs.append((scale, ox, oy))
    
    # Shuffle configs
    rng.shuffle(configs)
    
    for config_idx, (scale, ox, oy) in enumerate(configs):
        if time.time() - start_time > deadline:
            break
        
        # Generate base hexagonal lattice
        base_centers = _hexagonal_lattice(N, scale=scale, offset_x=ox, offset_y=oy)
        
        # Select free indices: ~12% of circles
        n_free = max(10, int(N * 0.12))
        free_indices = rng.choice(N, size=n_free, replace=False)
        
        # Add small random jitter to free circles
        jitter_scale = 0.02
        base_centers[free_indices] += rng.uniform(-jitter_scale, jitter_scale, size=(n_free, 2))
        base_centers = np.clip(base_centers, -0.5, 0.5)
        
        # Local polish
        polished = _local_polish(base_centers, free_indices, time_budget=3.0)
        
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
        
        # If we have time, do a second polish pass with different free set
        if time.time() - start_time < deadline - 10.0:
            free_indices2 = rng.choice(N, size=n_free, replace=False)
            polished2 = _local_polish(polished, free_indices2, time_budget=3.0)
            score2 = _score(polished2)
            if score2 > best_score:
                best_score = score2
                best_centers = polished2.copy()
        
        # Break early if we're close to deadline
        if time.time() - start_time > deadline - 15.0:
            break
    
    # Final polish pass on best solution
    if best_centers is not None and time.time() - start_time < deadline - 5.0:
        n_free_final = max(15, int(N * 0.15))
        free_indices_final = rng.choice(N, size=n_free_final, replace=False)
        best_centers = _local_polish(best_centers, free_indices_final, time_budget=8.0)
        final_score = _score(best_centers)
        if final_score > best_score:
            best_score = final_score
    
    # Ensure valid output
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
