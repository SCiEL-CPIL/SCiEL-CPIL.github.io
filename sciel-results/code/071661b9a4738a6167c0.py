# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(positions, a, y, w):
    """Compute the union area of the 64 triangles given base positions."""
    L = positions[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    return np.mean(union_lengths)

def _compute_area_and_subgrad(positions, a, y, w):
    """Compute area and subgradient simultaneously."""
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

def _subgradient_descent(positions, a, y, w, max_iter, deadline, initial_lr=0.01):
    """Subgradient descent to minimize the area."""
    n = len(positions)
    M = len(y)
    
    best_pos = positions.copy()
    best_area = _compute_area(best_pos, a, y, w)
    lr = initial_lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        area, grad = _compute_area_and_subgrad(positions, a, y, w)
        
        # Update
        positions = positions - lr * grad
        
        if area < best_area:
            best_area = area
            best_pos = positions.copy()
        
        # Decay learning rate
        lr *= 0.995
    
    return best_pos, best_area

def _coordinate_descent_polish(pos, a, y, w, deadline, step_sizes=None):
    """Coordinate descent with adaptive step sizes for fine-tuning."""
    n = len(pos)
    best_pos = pos.copy()
    best_area = _compute_area(best_pos, a, y, w)
    
    if step_sizes is None:
        step_sizes = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    
    for step in step_sizes:
        if time.time() > deadline - 30:
            break
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 30:
                break
            improved = False
            for idx in range(n):
                if time.time() > deadline - 30:
                    break
                # Try positive and negative steps
                for sign in [1, -1]:
                    x_new = best_pos.copy()
                    x_new[idx] += sign * step
                    area_new = _compute_area(x_new, a, y, w)
                    if area_new < best_area:
                        best_pos = x_new
                        best_area = area_new
                        improved = True
            if not improved:
                break
    
    return best_pos, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses Keich construction as starting point, then refines with subgradient descent,
    scipy optimization, and coordinate descent for fine-tuning.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 570.0  # 570 second limit with safety margin
    
    # Precompute fixed values
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    M = 20000
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    
    # Start with Keich construction
    pos = _keich_positions(n)
    best_pos = pos.copy()
    best_area = _compute_area(best_pos, a, y, w)
    
    # Phase 1: Coarse subgradient descent
    max_iter_phase1 = 3000
    pos, area = _subgradient_descent(pos, a, y, w, max_iter_phase1, deadline, initial_lr=0.05)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 60:
        return best_pos
    
    # Phase 2: Fine subgradient descent from best
    pos = best_pos.copy()
    max_iter_phase2 = 8000
    pos, area = _subgradient_descent(pos, a, y, w, max_iter_phase2, deadline, initial_lr=0.01)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 60:
        return best_pos
    
    # Phase 3: Scipy L-BFGS-B optimization
    if time.time() < deadline - 120:
        def objective(x):
            return _compute_area(x, a, y, w)
        
        def gradient(x):
            _, grad = _compute_area_and_subgrad(x, a, y, w)
            return grad
        
        try:
            result = minimize(objective, best_pos, method='L-BFGS-B', 
                             jac=gradient, options={'maxiter': 500, 'gtol': 1e-6})
            if result.fun < best_area:
                best_area = result.fun
                best_pos = result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 60:
        return best_pos
    
    # Phase 4: Coordinate descent fine-tuning
    step_sizes = [0.001, 0.0005, 0.0001, 0.00005]
    best_pos, best_area = _coordinate_descent_polish(best_pos, a, y, w, deadline, step_sizes)
    
    if time.time() > deadline - 30:
        return best_pos
    
    # Phase 5: Very fine coordinate descent
    step_sizes_fine = [0.00001, 0.000005, 0.000001]
    best_pos, best_area = _coordinate_descent_polish(best_pos, a, y, w, deadline, step_sizes_fine)
    
    return best_pos

# EVOLVE-BLOCK-END
