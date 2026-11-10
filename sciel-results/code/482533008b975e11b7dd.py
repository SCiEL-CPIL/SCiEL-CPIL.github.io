# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles given base positions x."""
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    y = (np.arange(n_heights) + 0.5) / n_heights  # heights y_m
    
    # L_i = x_i + a_i * y, w = (1-y)/n
    # Shape: (n, n_heights)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) / n  # Shape: (n_heights,)
    
    # Sort L at each height
    L_sorted = np.sort(L, axis=0)  # Shape: (n, n_heights)
    
    # Union length: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = np.diff(L_sorted, axis=0)  # Shape: (n-1, n_heights)
    union_lengths = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
    
    area = np.mean(union_lengths)
    return area


def _compute_area_soft(x, n=64, n_heights=20000, beta=50.0):
    """Compute area with soft-min for differentiability. Returns area and gradient."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    
    L = x[:, None] + a[:, None] * y[None, :]  # (n, n_heights)
    w = (1.0 - y[None, :]) / n  # (n_heights,)
    
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    
    # Soft-min: min(d, w) ≈ -log(exp(-beta*d) + exp(-beta*w)) / beta
    # But we need to be careful with the sorting - the soft approach is tricky
    # Let's use a different approach: compute the exact area and use subgradients
    
    # For now, use exact area
    union_lengths = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
    area = np.mean(union_lengths)
    
    # Compute subgradient with respect to x
    # The union length at height y is a piecewise-linear function of x
    # We can compute the subgradient by checking which parts of the min are active
    
    # For a simpler approach, use finite differences for gradient
    grad = np.zeros(n)
    eps = 1e-6
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        area_plus = _compute_area(x_plus, n, n_heights)
        grad[i] = (area_plus - area) / eps
    
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize base positions to minimize the area of union of 64 triangles.
    """
    start_time = time.time()
    deadline = start_time + 600  # 630s budget, leave 30s margin
    
    n = 64
    k = int(np.log2(n))
    
    # Initialize with Keich construction
    j = np.arange(1, k + 1)
    x = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    best_x = x.copy()
    best_area = _compute_area(x, n)
    
    # Gradient descent with adaptive step size
    lr = 0.01
    n_heights = 20000
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Compute area and gradient
        area, grad = _compute_area_soft(x, n, n_heights)
        
        # Update best
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Adaptive step size
        x_new = x - lr * grad
        
        # Check if step improved
        area_new = _compute_area(x_new, n, n_heights)
        
        if area_new < area:
            x = x_new
            lr = min(lr * 1.1, 0.1)  # Increase learning rate slightly
        else:
            lr = lr * 0.8  # Decrease learning rate
        
        # Periodically try larger steps
        if iteration % 100 == 0 and lr < 0.005:
            lr = 0.01
    
    return best_x


# EVOLVE-BLOCK-END
