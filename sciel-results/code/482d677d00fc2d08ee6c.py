# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(positions, a, y, w):
    """Compute the union area of the 64 triangles given base positions."""
    L = positions[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    return np.mean(union_lengths)

def _compute_area_and_grad(positions, a, y, w):
    """Compute area and subgradient simultaneously for efficiency."""
    n = len(positions)
    M = len(y)
    
    L = positions[None, :] + a[None, :] * y[:, None]  # (M, n)
    L_sorted = np.sort(L, axis=1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # (M, n-1)
    union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    area = np.mean(union_lengths)
    
    # Subgradient
    gap_lt_w = (diffs < w[:, None])  # (M, n-1) boolean
    grad_sorted = np.zeros_like(L_sorted)
    if n > 1:
        grad_sorted[:, 1:] += gap_lt_w
        grad_sorted[:, :-1] -= gap_lt_w
    
    perm = np.argsort(L, axis=1)  # (M, n)
    grad_original = np.zeros_like(L)
    rows = np.arange(M)[:, None]
    grad_original[rows, perm] = grad_sorted
    grad = np.mean(grad_original, axis=0)
    
    return area, grad

def _keich_positions(n=64):
    """Generate the Keich construction positions."""
    k = int(np.log2(n))
    positions = np.zeros(n, dtype=np.float64)
    j = np.arange(1, k + 1, dtype=np.float64)
    for i in range(n):
        eps = np.zeros(k, dtype=np.float64)
        for bit in range(k):
            if i & (1 << bit):
                eps[bit] = 1.0
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    return positions

def _subgradient_descent_momentum(positions, a, y, w, max_iter, deadline, 
                                    initial_lr=0.05, momentum=0.9):
    """Subgradient descent with Nesterov momentum."""
    n = len(positions)
    
    best_pos = positions.copy()
    best_area, _ = _compute_area_and_grad(best_pos, a, y, w)
    
    pos = positions.copy()
    vel = np.zeros(n, dtype=np.float64)
    lr = initial_lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Nesterov lookahead
        pos_look = pos - momentum * vel
        _, grad = _compute_area_and_grad(pos_look, a, y, w)
        
        vel = momentum * vel - lr * grad
        pos = pos + vel
        
        area, _ = _compute_area_and_grad(pos, a, y, w)
        if area < best_area:
            best_area = area
            best_pos = pos.copy()
        
        # Adaptive learning rate: decay slowly
        lr = initial_lr / (1.0 + 0.001 * it)
    
    return best_pos, best_area

def _coordinate_descent_polish(pos, a, y, w, deadline, max_passes=5):
    """Coordinate descent with line search for each coordinate."""
    n = len(pos)
    M = len(y)
    best_pos = pos.copy()
    best_area = _compute_area(best_pos, a, y, w)
    
    # Precompute for efficiency: for each coordinate i, 
    # L_i(y) = x_i + a_i * y, so changing x_i shifts all L_i by delta
    # We need to find optimal x_i given others fixed
    
    step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    
    for pass_num in range(max_passes):
        if time.time() > deadline - 30:
            break
        
        improved = False
        for idx in range(n):
            if time.time() > deadline - 30:
                break
            
            # For this coordinate, try to find best shift
            # Use ternary-like search around current value
            x_cur = best_pos[idx]
            best_x = x_cur
            best_a = best_area
            
            # Coarse search
            for step in step_sizes:
                for sign in [1, -1]:
                    x_try = x_cur + sign * step
                    new_pos = best_pos.copy()
                    new_pos[idx] = x_try
                    a_try = _compute_area(new_pos, a, y, w)
                    if a_try < best_a:
                        best_a = a_try
                        best_x = x_try
            
            if best_x != x_cur:
                best_pos[idx] = best_x
                best_area = best_a
                improved = True
        
        if not improved:
            break
    
    return best_pos, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses Keich construction, then refines with momentum subgradient descent
    and coordinate descent polishing.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Precompute fixed values
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    M = 20000
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    
    # Start with Keich construction
    pos = _keich_positions(n)
    best_pos = pos.copy()
    best_area = _compute_area(best_pos, a, y, w)
    
    # Phase 1: Coarse subgradient descent with momentum
    max_iter_phase1 = 3000
    pos, area = _subgradient_descent_momentum(pos, a, y, w, max_iter_phase1, deadline,
                                               initial_lr=0.08, momentum=0.9)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 100:
        return best_pos
    
    # Phase 2: Medium subgradient descent from best
    pos = best_pos.copy()
    max_iter_phase2 = 5000
    pos, area = _subgradient_descent_momentum(pos, a, y, w, max_iter_phase2, deadline,
                                               initial_lr=0.02, momentum=0.95)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 60:
        return best_pos
    
    # Phase 3: Fine subgradient descent
    pos = best_pos.copy()
    max_iter_phase3 = 3000
    pos, area = _subgradient_descent_momentum(pos, a, y, w, max_iter_phase3, deadline,
                                               initial_lr=0.005, momentum=0.98)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 30:
        return best_pos
    
    # Phase 4: Coordinate descent polish
    pos, area = _coordinate_descent_polish(best_pos, a, y, w, deadline, max_passes=3)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    return best_pos

# EVOLVE-BLOCK-END
