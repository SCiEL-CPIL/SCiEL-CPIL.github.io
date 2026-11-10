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
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    return area


def _line_search(positions, idx, n, n_heights, max_step=1.0, tol=1e-10, max_evals=30):
    """Binary line search along coordinate idx to find minimum area."""
    x0 = positions[idx]
    current_area = _compute_area(positions, n, n_heights)
    
    # First check if moving in either direction helps
    best_x = x0
    best_area = current_area
    
    for direction in [1.0, -1.0]:
        step = max_step
        for _ in range(max_evals):
            trial_x = x0 + direction * step
            positions[idx] = trial_x
            trial_area = _compute_area(positions, n, n_heights)
            
            if trial_area < best_area:
                best_x = trial_x
                best_area = trial_area
                # Keep going in same direction
                x0 = trial_x
                step *= 0.5
            else:
                # Binary search between best_x and trial_x
                lo = best_x
                hi = trial_x
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    positions[idx] = mid
                    mid_area = _compute_area(positions, n, n_heights)
                    if mid_area < best_area:
                        best_x = mid
                        best_area = mid_area
                        hi = mid
                    else:
                        lo = mid
                break
    
    positions[idx] = best_x
    return best_x, best_area


def _coordinate_descent(positions, n=64, max_rounds=100, n_heights=20000, 
                         max_step=1.0, tol=1e-12, deadline=None):
    """Coordinate descent with line search."""
    current = positions.copy().astype(np.float64)
    current_area = _compute_area(current, n, n_heights)
    best = current.copy()
    best_area = current_area
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for idx in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            old_x = current[idx]
            new_x, new_area = _line_search(current, idx, n, n_heights, 
                                            max_step=max_step, tol=tol)
            
            if abs(new_x - old_x) > 1e-14 and new_area < current_area - 1e-15:
                current[idx] = new_x
                current_area = new_area
                improved = True
                
                if current_area < best_area:
                    best = current.copy()
                    best_area = current_area
        
        if not improved:
            break
    
    return best, best_area


def _local_search(positions, n=64, max_iter=5000, step_size=0.01, n_heights=20000,
                  deadline=None):
    """Perform local search to minimize area."""
    current = positions.copy().astype(np.float64)
    current_area = _compute_area(current, n, n_heights)
    best = current.copy()
    best_area = current_area
    
    np.random.seed(42)
    
    for iter_idx in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random perturbations
        idx = np.random.randint(n)
        delta = np.random.uniform(-step_size, step_size)
        trial = current.copy()
        trial[idx] += delta
        
        trial_area = _compute_area(trial, n, n_heights)
        
        if trial_area < current_area:
            current = trial
            current_area = trial_area
            if current_area < best_area:
                best = current.copy()
                best_area = current_area
        
        # Gradually reduce step size
        step_size = max(0.001, step_size * 0.999)
    
    return best, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Returns positions that minimize the union area of the 64 triangles.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Start with the "aligned" initialization: x_i = -a_i + 0.5
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
    
    # Try several random initializations
    np.random.seed(123)
    for seed in range(10):
        np.random.seed(seed)
        random_pos = np.random.uniform(-1, 1, n)
        rand_area = _compute_area(random_pos, n, 20000)
        if rand_area < area1:
            positions = random_pos
            area1 = rand_area
    
    # Phase 1: Coarse local search
    best_positions, best_area = _local_search(positions, n, max_iter=2000, 
                                               step_size=0.1, n_heights=20000,
                                               deadline=deadline)
    
    # Phase 2: Coordinate descent for faster convergence
    if time.time() < deadline:
        best_positions, best_area = _coordinate_descent(best_positions, n, 
                                                        max_rounds=50, n_heights=20000,
                                                        max_step=0.5, deadline=deadline)
    
    # Phase 3: Fine local search
    if time.time() < deadline:
        best_positions, best_area = _local_search(best_positions, n, max_iter=3000, 
                                                   step_size=0.01, n_heights=20000,
                                                   deadline=deadline)
    
    # Phase 4: Final coordinate descent polish
    if time.time() < deadline:
        best_positions, best_area = _coordinate_descent(best_positions, n, 
                                                        max_rounds=100, n_heights=20000,
                                                        max_step=0.1, deadline=deadline)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
