# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the area of the union of 64 Kakeya-needle triangles.
    Uses analytical gradient for speed.
    """
    n = 64
    w = 1.0 / n
    a = np.arange(1, n + 1) / n  # a_i = (i+1)/64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    
    def compute_area_and_grad(x):
        """Compute the union area and its gradient with respect to x."""
        # L has shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort at each height and keep track of original indices
        idx = np.argsort(L, axis=1) # (M, n)
        L_sorted = np.take_along_axis(L, idx, axis=1)
        
        # Gaps between consecutive sorted values
        gaps = L_sorted[:, 1:] - L_sorted[:, :-1]  # shape (M, n-1)
        
        # Union length = w + sum(min(gaps, w))
        union_lengths = w + np.sum(np.minimum(gaps, w), axis=1)
        area = np.mean(union_lengths)
        
        # Gradient calculation
        # dU/dL_(k) for k=1..n
        # dU/dL_(k) = I[L_(k) - L_(k-1) < w] - I[L_(k+1) - L_(k) < w]
        
        dL_sorted = np.zeros_like(L_sorted)
        
        # mask_left[j] is True if gap j < w
        # gap j corresponds to L_sorted[:, j+1] - L_sorted[:, j]
        mask_left = gaps < w # (M, n-1)
        
        # If gap j < w, it contributes +1 to dL_sorted[:, j+1] and -1 to dL_sorted[:, j]
        dL_sorted[:, 1:] += mask_left
        dL_sorted[:, :-1] -= mask_left
        
        # Map back to original x indices
        # grad_x[m, i] = dL_sorted[m, idx[m, i]]
        grad_x = np.zeros_like(L)
        rows = np.arange(M)[:, None]
        cols = idx
        grad_x[rows, cols] = dL_sorted
        
        grad = np.mean(grad_x, axis=0)
        
        return area, grad
    
    # Start from the Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    deadline = time.time() + 580  # safety margin
    best_x = x0
    best_area = compute_area_and_grad(x0)[0]
    
    # Try L-BFGS-B with analytical gradient
    try:
        result = minimize(
            compute_area_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True, # Returns (f, g)
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success and result.fun < best_area:
            best_x = result.x
            best_area = result.fun
            
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the initial solution
        x_init = best_x + np.random.normal(0, 0.01, size=n)
        
        try:
            res = minimize(
                compute_area_and_grad,
                x_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success and res.fun < best_area:
                best_x = res.x
                best_area = res.fun
        except Exception:
            continue

    # Final validation
    best_x = np.asarray(best_x, dtype=np.float64)
    if not np.all(np.isfinite(best_x)):
        return x0
    
    return best_x

# EVOLVE-BLOCK-END
