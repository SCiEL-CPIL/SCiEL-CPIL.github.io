import numpy as np
import time


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area using quadrature."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
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
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    active = (diffs < w[np.newaxis, :]).astype(np.float64)  # (n-1, n_heights)
    
    # For each active gap between sorted positions j and j+1:
    #   subgrad contribution: +1 to the right triangle, -1 to the left triangle
    # At height m, the left triangle in sorted order at position j is sort_idx[j, m]
    # and the right triangle is sort_idx[j+1, m]
    
    # We need to scatter: for each (j, m) where active[j,m]=1:
    #   sg[sort_idx[j+1, m], m] += 1
    #   sg[sort_idx[j, m], m] -= 1
    
    # Vectorized approach using np.add.at
    sg = np.zeros((n, n_heights), dtype=np.float64)
    
    # Right triangles (sort_idx[1:, :])
    right_idx = sort_idx[1:, :]  # (n-1, n_heights)
    left_idx = sort_idx[:-1, :]  # (n-1, n_heights)
    
    # Flatten and use active as mask
    active_flat = active.ravel()  # (n*(n-1),)
    right_flat = right_idx.ravel()  # (n*(n-1),)
    left_flat = left_idx.ravel()  # (n*(n-1),)
    
    # Only process active entries
    mask = active_flat > 0.5
    if np.any(mask):
        r = right_flat[mask]
        l = left_flat[mask]
        # Need height indices
        # active has shape (n-1, n_heights), so element (j, m) has flat index j*n_heights + m
        # We need to scatter into sg[r, m] += 1 and sg[l, m] -= 1
        # Get the height index m for each active element
        # flat index = j * n_heights + m, so m = flat_idx % n_heights
        active_flat_indices = np.arange(active.size)[mask]
        m_indices = active_flat_indices % n_heights
        
        np.add.at(sg, (r, m_indices), 1.0)
        np.add.at(sg, (l, m_indices), -1.0)
    
    gradient = np.mean(sg, axis=1)
    
    return gradient


def _smooth_area_and_grad(positions, n=64, n_heights=20000, beta=50.0):
    """
    Smooth surrogate for the union area using soft-min.
    Union length at height y = w + sum_j min(d_j, w)
    min(d, w) = -softmax(-d, -w) * ... actually let's use:
    min(d_j, w) = d_j - max(0, d_j - w)
    max(0, x) ≈ (1/beta) * log(1 + exp(beta*x))  (softplus)
    
    So min(d_j, w) = d_j - softplus(beta*(d_j - w))/beta
    This is smooth and its gradient is well-defined.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    
    # softplus(beta * x) = log(1 + exp(beta*x)) / beta
    # For numerical stability:
    def softplus(x):
        # x can be large positive or negative
        # log(1 + exp(x)) for large x ≈ x
        # for large negative x ≈ 0
        return np.where(x > 20, x, np.log1p(np.exp(x)))
    
    z = beta * (diffs - w[np.newaxis, :])
    sp = softplus(z) / beta  # smooth max(0, d_j - w)
    
    # min(d_j, w) ≈ d_j - sp
    smooth_min = diffs - sp
    
    union_lengths = w + np.sum(smooth_min, axis=0)
    area = np.mean(union_lengths)
    
    # Gradient of area w.r.t. positions
    # d(area)/d(x_i) = (1/n_heights) * sum_m d(union_length_m)/d(x_i)
    # union_length_m = w_m + sum_j min(L_{(j+1),m} - L_{(j),m}, w_m)
    # d(min(d_j, w))/d(L_{(j+1)}) = 1 - sigmoid(beta*(d_j - w))  (since d(softplus)/dx = sigmoid)
    # d(min(d_j, w))/d(L_{(j)}) = -1 + sigmoid(beta*(d_j - w))
    
    sigmoid_z = 1.0 / (1.0 + np.exp(-z))  # (n-1, n_heights)
    
    # Contribution from each gap to each sorted position
    # For gap j at height m:
    #   d/dL_{(j+1)} = 1 - sigmoid_z[j,m]
    #   d/dL_{(j)} = -(1 - sigmoid_z[j,m]) = sigmoid_z[j,m] - 1
    
    # We need to map sorted positions back to original triangle indices
    sort_idx = np.argsort(L, axis=0)  # (n, n_heights)
    
    # Build gradient: for each original triangle i, sum contributions
    grad = np.zeros((n, n_heights), dtype=np.float64)
    
    # Right side (position j+1 in sorted order): contribution = 1 - sigmoid_z
    right_contrib = (1.0 - sigmoid_z)  # (n-1, n_heights)
    left_contrib = (sigmoid_z - 1.0)  # (n-1, n_heights)
    
    # Scatter using sort_idx
    # For each (j, m): right triangle is sort_idx[j+1, m], gets right_contrib[j, m]
    # For each (j, m): left triangle is sort_idx[j, m], gets left_contrib[j, m]
    
    right_idx = sort_idx[1:, :]  # (n-1, n_heights)
    left_idx = sort_idx[:-1, :]  # (n-1, n_heights)
    
    right_flat = right_idx.ravel()
    left_flat = left_idx.ravel()
    rc_flat = right_contrib.ravel()
    lc_flat = left_contrib.ravel()
    
    n_total = (n - 1) * n_heights
    m_indices = np.arange(n_total) % n_heights
    
    np.add.at(grad, (right_flat, m_indices), rc_flat)
    np.add.at(grad, (left_flat, m_indices), lc_flat)
    
    gradient = np.mean(grad, axis=1)
    
    return area, gradient


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize Kakeya needle triangle positions using smooth surrogate + subgradient.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    a = (np.arange(n) + 1) / n
    
    # Initialize with the known good construction
    x0 = -a + 0.5
    
    # Phase 1: Smooth optimization with adaptive learning rate
    current = x0.copy()
    current_area = _compute_area(current, n, 20000)
    best_positions = current.copy()
    best_area = current_area
    
    beta_schedule = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
    lr_start = 0.05
    
    for beta in beta_schedule:
        if time.time() > deadline - 60:
            break
        
        lr = lr_start
        n_iters = 500
        
        for it in range(n_iters):
            if time.time() > deadline - 60:
                break
            
            area_s, grad = _smooth_area_and_grad(current, n, 20000, beta)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            trial = current - lr * grad
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
                # Small increase in lr on success
                lr = min(lr * 1.02, 0.1)
            else:
                # Decrease lr on failure
                lr *= 0.95
            
            # Adaptive lr decay
            lr = max(lr * 0.9995, 0.001)
    
    # Phase 2: Pure subgradient descent with momentum
    if time.time() < deadline - 30:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.02
        momentum = np.zeros(n)
        beta_mom = 0.9
        
        n_iters = 2000
        
        for it in range(n_iters):
            if time.time() > deadline - 30:
                break
            
            grad = _subgradient_vec(current, n, 20000)
            
            # Normalize
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            momentum = beta_mom * momentum - lr * grad
            trial = current + momentum
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
                lr = min(lr * 1.01, 0.05)
            else:
                momentum *= 0.5
                lr *= 0.97
            
            lr = max(lr * 0.9998, 0.002)
    
    # Phase 3: Fine local search
    if time.time() < deadline - 5:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.005
        
        np.random.seed(42)
        n_iters = 3000
        
        for it in range(n_iters):
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
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.0005, step_size * 0.9995)
    
    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
