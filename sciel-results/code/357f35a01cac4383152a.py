import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    w = 1.0 / n
    a = (np.arange(1, n + 1)) / n  # a_i for i=0..63
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M  # heights
    
    # Keich construction as initial guess
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    # Compute area and subgradient
    def compute_area_and_grad(x):
        # L_i at each height: L = x_i + a_i * y
        # x has shape (n,), a has shape (n,), y_vals has shape (M,)
        # L has shape (M, n)
        L = x[None, :] + a[None, :] * y_vals[:, None]
        
        # Sort L along axis 1
        sort_idx = np.argsort(L, axis=1)
        L_sorted = np.take_along_axis(L, sort_idx, axis=1)
        
        # Union length at each height
        # U = w + sum_{j=0}^{n-2} min(L_sorted[:, j+1] - L_sorted[:, j], w)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # shape (M, n-1)
        U = w + np.sum(np.minimum(diffs, w), axis=1)  # shape (M,)
        area = np.mean(U)
        
        # Subgradient w.r.t. x_i
        # For each height, the subgradient w.r.t. L_{(j)} is:
        #   -1 if j > 0 and L_{(j)} - L_{(j-1)} < w
        #   +1 if j < n-1 and L_{(j+1)} - L_{(j)} < w
        #   (sum of these contributions)
        
        # grad_L has shape (M, n)
        grad_L = np.zeros_like(L)
        # Left contribution: for j >= 1, if diff[j-1] < w, grad_L[:, j] += -1
        left_diffs = diffs  # diffs[:, j] = L_sorted[:, j+1] - L_sorted[:, j] for j=0..n-2
        # For position j in sorted order (0-indexed):
        # Contribution from right neighbor (j < n-1): +1 if diffs[:, j] < w
        # Contribution from left neighbor (j > 0): -1 if diffs[:, j-1] < w
        
        # For j=0: only right contribution
        grad_L[:, 0] = np.where(diffs[:, 0] < w, 1.0, 0.0)
        # For j=n-1: only left contribution
        grad_L[:, n-1] = np.where(diffs[:, n-2] < w, -1.0, 0.0)
        # For 0 < j < n-1:
        if n > 2:
            grad_L[:, 1:n-1] = (np.where(diffs[:, 1:] < w, 1.0, 0.0) + 
                                np.where(diffs[:, :-1] < w, -1.0, 0.0))
        
        # Now map back from sorted order to original order
        # grad_x has shape (n,)
        # For each height m, grad_L[m, sort_idx[m, i]] gives the subgradient w.r.t. L_i at height m
        # The subgradient w.r.t. x_i is the mean over heights of grad_L[m, i] (since dL_i/dx_i = 1)
        
        # Actually, we need to unsort: for each m, the gradient w.r.t. x_i is grad_L[m, pos] where pos is the position of i in the sorted order
        # Equivalent: grad_x[i] = mean_m grad_L[m, sort_idx[m, i]] ... no wait
        
        # Let me think again. L_sorted[m, j] = L[m, sort_idx[m, j]]
        # grad_L[m, j] is the subgradient w.r.t. L_sorted[m, j]
        # We want grad w.r.t. L[m, i], which is grad_L[m, j] where sort_idx[m, j] = i
        # So: grad_L_orig[m, i] = grad_L[m, j] where j = inverse_sort[m, i]
        
        # Create inverse sort index
        inv_sort = np.argsort(sort_idx, axis=1)  # inv_sort[m, i] = position of i in sorted order
        
        # Actually, simpler: 
        # grad_L_orig = np.zeros_like(L)
        # np.put_along_axis(grad_L_orig, sort_idx, grad_L, axis=1)
        
        grad_L_orig = np.zeros_like(L)
        np.put_along_axis(grad_L_orig, sort_idx, grad_L, axis=1)
        
        # grad_x[i] = mean over m of grad_L_orig[m, i]
        grad_x = np.mean(grad_L_orig, axis=0)
        
        return area, grad_x
    
    # Subgradient descent with adaptive step size
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit, use 580 for safety
    
    best_x = x.copy()
    best_area, _ = compute_area_and_grad(x)
    
    # Use Polyak-like step size or fixed decreasing
    # First, get a rough idea of the gradient magnitude
    _, g0 = compute_area_and_grad(x)
    g_norm = np.linalg.norm(g0)
    
    # Use a learning rate schedule
    lr = 0.01
    best_lr = lr
    momentum = 0.9
    v = np.zeros(n)
    
    iter_count = 0
    while time.time() < deadline - 30:  # 30s safety margin
        area, grad = compute_area_and_grad(x)
        iter_count += 1
        
        # Momentum update
        v = momentum * v - lr * grad
        x_new = x + v
        
        # Check if improved
        area_new, _ = compute_area_and_grad(x_new)
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
        
        # Adaptive learning rate
        if iter_count % 100 == 0:
            # Reduce learning rate if no improvement
            if area_new >= best_area * 0.999:
                lr *= 0.9
            # Try to increase if making good progress
            elif area_new < best_area * 0.995:
                lr = min(lr * 1.05, 0.1)
        
        x = x_new
    
    return best_x
