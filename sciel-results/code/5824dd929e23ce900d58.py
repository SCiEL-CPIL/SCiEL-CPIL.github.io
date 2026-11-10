import numpy as np
import time

# Precompute constants for area evaluation
_N = 64
_N_HEIGHTS = 20000
_A = (np.arange(_N) + 1) / _N
_Y = (np.arange(_N_HEIGHTS) + 0.5) / _N_HEIGHTS
_W = (1 - _Y) / _N


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area using quadrature."""
    a = _A[:n]
    y = _Y[:n_heights]
    w = _W[:n_heights]
    
    # L_i = x_i + a_i * y for each triangle i and height m
    # Shape: (n, n_heights)
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L at each height
    L_sorted = np.sort(L, axis=0)  # (n, n_heights)
    
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    overlaps = np.minimum(diffs, w[np.newaxis, :])  # (n-1, n_heights)
    
    union_lengths = w + np.sum(overlaps, axis=0)  # (n_heights,)
    area = np.mean(union_lengths)
    
    return area


def _line_search(current, idx, n, n_heights, max_iter=20):
    """Perform line search along dimension idx to minimize area."""
    x0 = current[idx]
    lo, hi = -10.0, 10.0
    
    for _ in range(max_iter):
        mid1 = lo + (hi - lo) / 3.0
        mid2 = hi - (hi - lo) / 3.0
        
        trial1 = current.copy()
        trial1[idx] = mid1
        area1 = _compute_area(trial1, n, n_heights)
        
        trial2 = current.copy()
        trial2[idx] = mid2
        area2 = _compute_area(trial2, n, n_heights)
        
        if area1 < area2:
            hi = mid2
        else:
            lo = mid1
    
    return (lo + hi) / 2.0


def _coordinate_descent(positions, n=64, max_iter=500, n_heights=20000, tol=1e-8):
    """Perform coordinate descent to minimize area."""
    current = positions.copy().astype(np.float64)
    current_area = _compute_area(current, n, n_heights)
    best = current.copy()
    best_area = current_area
    
    for iter_idx in range(max_iter):
        improved = False
        for idx in range(n):
            new_val = _line_search(current, idx, n, n_heights)
            if abs(new_val - current[idx]) > 1e-10:
                trial = current.copy()
                trial[idx] = new_val
                trial_area = _compute_area(trial, n, n_heights)
                if trial_area < current_area - tol:
                    current = trial
                    current_area = trial_area
                    improved = True
                    if current_area < best_area:
                        best = current.copy()
                        best_area = current_area
        
        if not improved:
            break
    
    return best, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Returns positions that minimize the union area of the 64 triangles.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill
    
    # Start with the "aligned" initialization: x_i = -a_i + 0.5
    # This centers the triangles and should give good overlap
    a = (np.arange(n) + 1) / n
    positions = -a + 0.5
    
    # Also try the Keich construction as an alternative start
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    keich_positions = np.array(keich_positions, dtype=np.float64)
    
    # Evaluate both starting points
    area1 = _compute_area(positions, n, 20000)
    area2 = _compute_area(keich_positions, n, 20000)
    
    if area2 < area1:
        positions = keich_positions
        area1 = area2
    
    # Try a few random initializations
    np.random.seed(123)
    for _ in range(5):
        random_pos = np.random.uniform(-1, 1, n)
        rand_area = _compute_area(random_pos, n, 20000)
        if rand_area < area1:
            positions = random_pos
            area1 = rand_area
    
    # Perform coordinate descent (exploits piecewise-linear structure)
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        best_positions, best_area = _coordinate_descent(positions, n, max_iter=100, n_heights=20000)
    
    # Refine with random local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        np.random.seed(42)
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.05
        max_iter = 2000
        for iter_idx in range(max_iter):
            if time.time() > deadline - 5:
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
                    best_positions = current.copy()
                    best_area = current_area
            
            step_size = max(0.001, step_size * 0.999)
    
    # Final coordinate descent refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        best_positions, best_area = _coordinate_descent(best_positions, n, max_iter=50, n_heights=20000)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
