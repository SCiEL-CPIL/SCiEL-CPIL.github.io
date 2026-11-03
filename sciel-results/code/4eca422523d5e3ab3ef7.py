"""Improved packing: greedy candidate selection on hexagonal lattice + local refinement."""
import numpy as np
import time

N = 200
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _compute_radius(centres):
    """Compute the radius for a set of centres (must be inside [-0.5,0.5]^2)."""
    if len(centres) == 0:
        return 0.0
    # Boundary constraint
    dist_to_boundary = np.min(0.5 - np.abs(centres), axis=1)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(dist_to_boundary))


def _build_hex_lattice(resolution=60):
    """Build a hexagonal lattice of candidate points inside the square."""
    # Hexagonal lattice spacing
    # For a hex grid, we use two basis vectors
    # a1 = (s, 0), a2 = (s/2, s*sqrt(3)/2)
    # We want points inside [-0.5, 0.5]^2
    
    # Choose spacing so we get a good number of candidates
    # Let's aim for ~5000-20000 candidates
    s = 1.0 / resolution  # spacing
    
    points = []
    # Generate hex lattice
    # For each row, offset alternates
    n_rows = int(1.0 / (s * np.sqrt(3) / 2)) + 2
    n_cols = int(1.0 / s) + 2
    
    for i in range(-n_rows, n_rows + 1):
        y = i * s * np.sqrt(3) / 2
        if y > 0.5 or y < -0.5:
            continue
        offset = 0.0 if i % 2 == 0 else s / 2
        for j in range(-n_cols, n_cols + 1):
            x = j * s + offset
            if x <= 0.5 and x >= -0.5 and y <= 0.5 and y >= -0.5:
                points.append([x, y])
    
    return np.array(points)


def _greedy_select(candidates, n_select):
    """Greedy farthest-point sampling to select n_select points from candidates."""
    n_cand = len(candidates)
    selected = []
    remaining = set(range(n_cand))
    
    # Start with a random point (or center)
    # Use the center of the square as initial
    center_idx = np.argmin(np.sum(candidates ** 2, axis=1))
    selected.append(center_idx)
    remaining.discard(center_idx)
    
    # Maintain min distance to any selected point for each candidate
    min_dists = np.full(n_cand, np.inf)
    
    for _ in range(n_select - 1):
        # Update min_dists for newly added point
        new_pt = candidates[selected[-1]]
        d = np.sqrt(np.sum((candidates - new_pt) ** 2, axis=1))
        min_dists = np.minimum(min_dists, d)
        
        # Also consider boundary distance
        # For each candidate, the effective "distance" is min(min_dist_to_selected, dist_to_boundary)
        # Actually, for greedy, we want to maximize the minimum of:
        #   - distance to nearest selected point
        #   - distance to boundary
        # So the score for a candidate is min(min_dists[i], boundary_dist[i])
        
        boundary_dists = np.min(0.5 - np.abs(candidates), axis=1)
        scores = np.minimum(min_dists, boundary_dists)
        
        # Set non-remaining to -inf
        for idx in selected:
            scores[idx] = -np.inf
        
        # Pick the best remaining
        best_idx = int(np.argmax(scores))
        if best_idx not in remaining:
            # Find best in remaining
            for idx in remaining:
                if scores[idx] > -np.inf:
                    best_idx = idx
                    break
            else:
                break
        
        selected.append(best_idx)
        remaining.discard(best_idx)
    
    return candidates[selected]


def _local_refinement(centres, n_iter=500, step_size=0.01):
    """Simple local refinement using random perturbations."""
    best = centres.copy()
    best_r = _compute_radius(best)
    
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        # Try moving a random point
        i = np.random.randint(len(centres))
        orig = best[i].copy()
        # Random small perturbation
        best[i] = orig + np.random.randn(2) * step_size
        
        # Check if still valid (inside square)
        if np.any(np.abs(best[i]) > 0.5):
            best[i] = orig
            continue
        
        new_r = _compute_radius(best)
        if new_r > best_r:
            best_r = new_r
        else:
            best[i] = orig
    
    return best


def construct():
    """Construct 200 circle centres in [-0.5, 0.5]^2 maximizing minimum radius."""
    t0 = time.time()
    
    # Build candidate lattice
    candidates = _build_hex_lattice(resolution=50)
    
    # Greedy selection
    selected = _greedy_select(candidates, N)
    
    # Local refinement
    if time.time() - t0 < TIME_LIMIT - 5:
        selected = _local_refinement(selected, n_iter=300, step_size=0.005)
    
    # Ensure valid
    selected = np.clip(selected, -0.5, 0.5)
    
    return selected
