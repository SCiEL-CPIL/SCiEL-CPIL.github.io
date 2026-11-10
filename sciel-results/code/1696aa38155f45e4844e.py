# EVOLVE-BLOCK-START
import numpy as np
import time


def _keich_positions(n=64):
    """Classical Keich construction for n a power of 2."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _union_area(x, n=64, n_heights=20000):
    """Compute the union area of the n triangles using quadrature at n_heights levels."""
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    w = 1.0 / n
    # Heights y_m = (m + 0.5) / n_heights
    m = np.arange(n_heights)
    y = (m + 0.5) / n_heights  # shape (n_heights,)
    
    # L_i(y) = x_i + a_i * y, shape (n_heights, n)
    L = x[None, :] + a[None, :] * y[:, None]
    
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)
    
    # Union length at each height: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = np.diff(L_sorted, axis=1)  # shape (n_heights, n-1)
    union_lengths = w + np.sum(np.minimum(diffs, w), axis=1)  # shape (n_heights,)
    
    area = np.mean(union_lengths)
    return area


def _grad_descent(x0, n=64, lr=0.01, max_iter=500, tol=1e-10):
    """Simple gradient descent with numerical gradients."""
    x = x0.copy()
    best_x = x.copy()
    best_area = _union_area(x, n)
    
    for it in range(max_iter):
        # Compute numerical gradient
        grad = np.zeros_like(x)
        eps = 1e-5
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (_union_area(x_plus, n) - _union_area(x_minus, n)) / (2 * eps)
        
        # Update
        x_new = x - lr * grad
        area_new = _union_area(x_new, n)
        
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
        
        if abs(best_area - area_new) < tol:
            break
        x = x_new
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Uses Keich construction as starting point, then optimizes with gradient descent.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with Keich construction
    x = _keich_positions(n)
    best_x = x.copy()
    best_area = _union_area(x, n)
    
    # Try multiple learning rates and restarts
    lrs = [0.05, 0.02, 0.01, 0.005, 0.001]
    
    for lr in lrs:
        if time.time() > deadline - 60:  # 60s safety margin
            break
        
        x_current = _keich_positions(n).copy()
        
        # Gradient descent
        for it in range(2000):
            if time.time() > deadline - 30:
                break
            
            # Numerical gradient
            eps = 1e-4
            grad = np.zeros(n)
            for i in range(n):
                xp = x_current.copy()
                xp[i] += eps
                xm = x_current.copy()
                xm[i] -= eps
                grad[i] = (_union_area(xp, n) - _union_area(xm, n)) / (2 * eps)
            
            # Adaptive step size
            step = lr / (1 + it * 0.01)
            x_new = x_current - step * grad
            
            area_new = _union_area(x_new, n)
            
            if area_new < best_area:
                best_area = area_new
                best_x = x_new.copy()
            
            if area_new < _union_area(x_current, n):
                x_current = x_new
            else:
                # Reduce learning rate
                lr *= 0.95
            
            if it % 100 == 0:
                pass  # Could add early stopping
        
        # Restart from best found so far
        x_current = best_x.copy()
    
    # Final polish with small steps
    x_current = best_x.copy()
    lr_small = 1e-4
    for it in range(500):
        if time.time() > deadline - 10:
            break
        
        eps = 1e-5
        grad = np.zeros(n)
        for i in range(n):
            xp = x_current.copy()
            xp[i] += eps
            xm = x_current.copy()
            xm[i] -= eps
            grad[i] = (_union_area(xp, n) - _union_area(xm, n)) / (2 * eps)
        
        x_new = x_current - lr_small * grad
        area_new = _union_area(x_new, n)
        
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
            x_current = x_new
        else:
            lr_small *= 0.9
    
    return best_x


# EVOLVE-BLOCK-END
