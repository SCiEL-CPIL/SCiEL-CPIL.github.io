# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _keich_init(n):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def _area_and_grad(x, n=64, M=20000):
    # x: shape (n,)
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    # L_i(y) = x_i + a_i * y_m
    # L shape: (M, n)
    L = x[None, :] + a[None, :] * y[:, None]
    
    # Sort L at each height
    # idx shape: (M, n)
    idx = np.argsort(L, axis=1)
    # L_sorted shape: (M, n)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    
    # Gaps D_k = L_sorted[:, k+1] - L_sorted[:, k]
    # D shape: (M, n-1)
    D = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # Indicator: D < w
    # w shape: (M, 1)
    # D < w[:, None]
    overlap = D < w[:, None]
    
    # Gradient contribution for each sorted position j
    # g_j = I(D_{j-1} < w) - I(D_j < w)
    # For j=0: g_0 = -I(D_0 < w)  (D_{-1} is 0)
    # For j=n-1: g_{n-1} = I(D_{n-2} < w) (D_{n-1} is 0)
    # For 0 < j < n-1: g_j = I(D_{j-1} < w) - I(D_j < w)
    
    # Create a zero array for gradients in sorted order
    g_sorted = np.zeros_like(L_sorted)
    
    # Subtract I(D_j < w) from position j+1
    # overlap is (M, n-1), corresponds to gaps between j and j+1
    # So for gap k (between sorted k and k+1), it contributes -1 to sorted k+1 and +1 to sorted k
    # g_sorted[:, k] += overlap[:, k]
    # g_sorted[:, k+1] -= overlap[:, k]
    
    g_sorted[:, :-1] += overlap
    g_sorted[:, 1:] -= overlap
    
    # Now g_sorted[:, j] is the subgradient of U(y_m) w.r.t L_sorted[:, j]
    # We need to map this back to x_i
    # grad_x[i] = a_i * sum_m g_sorted[m, idx[m, i]] / M
    
    # We can use np.add.at or scatter
    # Let's use a loop over M? No, vectorize.
    # We have idx (M, n) and g_sorted (M, n).
    # We want to sum g_sorted[m, idx[m, i]] for each i.
    # This is equivalent to: for each m, add g_sorted[m] to grad_x at indices idx[m].
    
    grad_x = np.zeros(n)
    # Using a loop over M might be slow if M is large, but 20000 is okay in Python if optimized?
    # Better: use np.bincount or similar?
    # Actually, we can just loop over i? No.
    # Let's try to vectorize the inverse permutation.
    # inv_idx[m, j] = i such that idx[m, i] = j.
    # Then grad_x[i] = sum_m g_sorted[m, inv_idx[m, i]] * a_i / M
    
    # Compute inverse permutation
    inv_idx = np.zeros_like(idx)
    # np.put_along_axis?
    # Or simply:
    # For each m, inv_idx[m, idx[m, i]] = i
    # This is tricky to vectorize directly without advanced indexing.
    # Let's use the fact that idx is a permutation.
    # We can use:
    # grad_x_unscaled = np.zeros(n)
    # for m in range(M):
    #    grad_x_unscaled += g_sorted[m, idx[m]]  # This adds g_sorted[m, i] to grad_x_unscaled[i]? No.
    #    # g_sorted[m, k] is the gradient for the k-th smallest L.
    #    # The k-th smallest L corresponds to original index idx[m, k].
    #    # So we want to add g_sorted[m, k] to grad_x_unscaled[idx[m, k]].
    #    np.add.at(grad_x_unscaled, idx[m], g_sorted[m])
    
    # 20000 iterations of np.add.at might be slow.
    # Alternative:
    # Create a matrix G of shape (M, n) where G[m, i] = g_sorted[m, inv_idx[m, i]]
    # We can do this by:
    # G = np.zeros_like(L)
    # np.put_along_axis(G, idx, g_sorted, axis=1)
    # Then grad_x = np.sum(G, axis=0) * a / M
    
    G = np.zeros_like(L)
    np.put_along_axis(G, idx, g_sorted, axis=1)
    grad_x = np.sum(G, axis=0) * a / M
    
    # Area calculation
    # U_m = w_m + sum_k min(D_k, w_m)
    # sum_k min(D_k, w_m) = sum_k D_k * I(D_k < w_m) + sum_k w_m * I(D_k >= w_m)
    # = sum_k D_k * overlap + (n-1) * w_m - sum_k D_k * (1 - overlap) ?
    # Simpler:
    # min(D, w) = D - max(D-w, 0)
    # Or just compute directly.
    min_terms = np.where(overlap, D, w[:, None])
    U = w + np.sum(min_terms, axis=1)
    area = np.mean(U)
    
    return area, grad_x

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x0 = _keich_init(n)
    
    # Check time
    start_time = time.time()
    deadline = start_time + 600
    
    # Initial area
    area0, _ = _area_and_grad(x0, n)
    
    # Optimize
    # We use L-BFGS-B with the subgradient as jac
    # Maximize iterations but stop if time runs out
    # L-BFGS-B doesn't have a direct time limit, so we wrap the objective
    # to check time and raise an exception or return early?
    # Better: run minimize with a custom callback or just check time after?
    # Minimize will run for a fixed number of iterations.
    # We can estimate time per iteration.
    
    # Let's try a few iterations.
    # One call to _area_and_grad takes ~0.1s?
    # Let's measure.
    t1 = time.time()
    _area_and_grad(x0, n)
    t2 = time.time()
    time_per_eval = t2 - t1
    
    # L-BFGS-B does multiple function evaluations per iteration.
    # Let's allow 500 iterations?
    # If time_per_eval is 0.1s, 500 iters * 10 evals/iter = 50s.
    # We have 600s.
    
    # However, if the subgradient is noisy or the function is flat, it might not improve.
    
    res = minimize(_area_and_grad, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': 1000, 'maxfun': 10000})
    
    # Check if we have time to do more?
    # If res.success is false due to maxiter, we can try again with more?
    # Or just return res.x
    
    # Ensure finite
    if not np.all(np.isfinite(res.x)):
        return x0
    
    return res.x

# EVOLVE-BLOCK-END
