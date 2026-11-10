import numpy as np
import time


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area using quadrature."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    return area


def _subgradient_vec(positions, n=64, n_heights=20000):
    """
    Vectorized subgradient of the union area with respect to positions.
    Union length at height y = w + sum_{j=0}^{n-2} min(d_j, w)
    where d_j = L_{(j+1)} - L_{(j)} (sorted).
    Subgradient of min(d_j, w):
      -1 on L_{(j)}, +1 on L_{(j+1)} if d_j < w
      0 if d_j > w
    Since dL_i/dx_i = 1, we accumulate per-triangle contributions.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort and track original indices
    sort_idx = np.argsort(L, axis=0)  # (n, n_heights)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    
    # Mask: True where gap < w (overlapping pair)
    mask = diffs < w[np.newaxis, :]  # (n-1, n_heights)
    
    # For each overlapping pair j (sorted position j and j+1):
    #   +1 to the triangle at sorted position j+1
    #   -1 to the triangle at sorted position j
    # We need to scatter these into the original-triangle index space.
    
    # right_idx[m, j] = original triangle at sorted position j+1, height m
    # left_idx[m, j] = original triangle at sorted position j, height m
    right_idx = sort_idx[1:, :]  # (n-1, n_heights)
    left_idx = sort_idx[:-1, :]  # (n-1, n_heights)
    
    # Build contribution array (n, n_heights)
    contrib = np.zeros((n, n_heights), dtype=np.float64)
    
    # Scatter +1 where mask is True, at right_idx positions
    # and -1 at left_idx positions
    # Use np.add.at for unbuffered scatter-add
    np.add.at(contrib, (right_idx, np.broadcast_to(np.arange(n_heights), (n-1, n_heights))),
              np.where(mask, 1.0, 0.0))
    np.add.at(contrib, (left_idx, np.broadcast_to(np.arange(n_heights), (n-1, n_heights))),
              np.where(mask, -1.0, 0.0))
    
    gradient = np.mean(contrib, axis=1)
    return gradient


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Direct construction for Kakeya needle triangle positions.
    
    Uses the structural insight that triangles with larger a_i tilt more,
    so we offset their base positions to maximize overlap at all heights.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    a = (np.arange(n) + 1) / n
    
    # Try multiple initializations
    candidates = []
    
    # Aligned construction
    candidates.append(-a + 0.5)
    
    # Various c values
    for c_try in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        candidates.append(-a * c_try + 0.5 * c_try)
    
    # Random perturbations of aligned
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        pert = np.random.normal(0, 0.05, n)
        candidates.append(-a + 0.5 + pert)
    
    # Evaluate all candidates
    best_positions = None
    best_area = np.inf
    
    for pos_try in candidates:
        if time.time() > deadline:
            break
        area_try = _compute_area(pos_try, n, 20000)
        if area_try < best_area:
            best_area = area_try
            best_positions = pos_try.copy()
    
    # Gradient descent phase using vectorized subgradient
    if time.time() < deadline - 60:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.05
        
        for it in range(20000):
            if time.time() > deadline - 60:
                break
            
            grad = _subgradient_vec(current, n, 20000)
            current = current - lr * grad
            current_area = _compute_area(current, n, 20000)
            
            if current_area < best_area:
                best_area = current_area
                best_positions = current.copy()
            
            # Adaptive learning rate
            if it % 100 == 0:
                lr = max(0.001, lr * 0.95)
    
    # Local search phase 1: coarse
    if time.time() < deadline - 30:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.03
        
        np.random.seed(42)
        for it in range(5000):
            if time.time() > deadline - 30:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.005, step_size * 0.9997)
    
    # Local search phase 2: fine
    if time.time() < deadline - 15:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.01
        
        np.random.seed(123)
        for it in range(4000):
            if time.time() > deadline - 15:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.001, step_size * 0.9995)
    
    # Gradient polish phase
    if time.time() < deadline - 5:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.01
        
        for it in range(5000):
            if time.time() > deadline - 5:
                break
            
            grad = _subgradient_vec(current, n, 20000)
            current = current - lr * grad
            current_area = _compute_area(current, n, 20000)
            
            if current_area < best_area:
                best_area = current_area
                best_positions = current.copy()
            
            lr *= 0.999
    
    # Final micro-refinement
    if time.time() < deadline - 3:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.002
        
        np.random.seed(999)
        for it in range(1500):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.0005, step_size * 0.999)
    
    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
