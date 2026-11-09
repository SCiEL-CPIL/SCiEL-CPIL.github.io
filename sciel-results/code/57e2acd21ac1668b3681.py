import numpy as np
import time

def _compute_scores(polygon):
    """Compute per-vertex scores and return the minimum."""
    n = len(polygon)
    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    # Compute cross products of consecutive edges
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(cross == 0):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        dists = np.linalg.norm(normalized - normalized[i], axis=1)
        dists = np.sort(dists[1:])  # exclude self
        if np.any(dists < 0.05):
            return 0.0
        gaps = np.diff(dists)
        k = np.argmin(gaps)
        D = (dists[k] + dists[k+1]) / 2
        if D < 0.1:
            return 0.0
        if k == 0:
            selected = dists[0:4]
        elif k == 17:
            selected = dists[15:19]
        else:
            selected = dists[k-1:k+3]
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
        if min_score == 0.0:
            return 0.0
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    # Start with regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    polygon = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _compute_scores(polygon)
    best_polygon = polygon.copy()
    
    # Gradient-free optimization: perturb vertices and keep improvements
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() < deadline - 10.0:
        # Try random perturbations
        improved = False
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            trial = best_polygon + np.random.randn(n, 2) * step_size
            score = _compute_scores(trial)
            if score > best_score + 1e-10:
                best_score = score
                best_polygon = trial
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_polygon
