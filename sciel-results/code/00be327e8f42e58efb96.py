# EVOLVE-BLOCK-START
import numpy as np
import time

def _union_area_fast(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    Optimized for speed: vectorized operations, minimal Python loops.
    """
    # Pre-compute constants
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    # L_i(y) = x_i + a_i * y
    # Shape: (n, M)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0
    # argsort is O(n log n * M), which is the bottleneck.
    # For n=64, this is fast.
    L_sorted = np.sort(L, axis=0)
    
    # Consecutive differences
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length = w + sum over j of min(d_j, w)
    min_d_w = np.minimum(d, w[np.newaxis, :])
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # 1. Generate initial candidates
    # Keich construction is a good starting point
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0_keich[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    # Random initialization
    x0_rand = np.random.uniform(-1.0, 1.0, n)
    
    # Sorted uniform
    x0_sorted = np.linspace(-1.0, 1.0, n)
    
    candidates = [x0_keich, x0_rand, x0_sorted]
    
    best_x = None
    best_area = np.inf
    
    # Evaluate initial candidates
    for x0 in candidates:
        area = _union_area_fast(x0)
        if area < best_area:
            best_area = area
            best_x = x0.copy()
    
    # 2. Simulated Annealing
    # The objective is piecewise linear. SA can jump over the kinks.
    current_x = best_x.copy()
    current_area = best_area
    
    # Initial temperature
    T = 0.1
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Step size for perturbations
    step_size = 0.5
    
    # We will run SA until time runs out or T < T_min
    # To speed up, we can batch the evaluation if we were using a different metric,
    # but here we evaluate one candidate at a time.
    
    # Optimization: Pre-allocate arrays if possible? 
    # _union_area_fast is already vectorized.
    
    # To make SA more effective, we use a "restart" strategy:
    # If we get stuck, we perturb more aggressively.
    
    iterations = 0
    while time.time() < deadline - 10.0:
        if T < T_min:
            break
            
        # Generate perturbation
        # Use Gaussian noise scaled by T and step_size
        # We want to explore the space.
        # A single coordinate update is too slow. Global perturbation is better for SA.
        # However, large global perturbations might break good structures.
        # Let's try a mix: small global + occasional large local.
        
        # Perturb all coordinates
        noise = np.random.randn(n) * step_size * np.sqrt(T)
        x_new = current_x + noise
        
        # Evaluate
        area_new = _union_area_fast(x_new)
        
        # Metropolis criterion
        delta_area = area_new - current_area
        if delta_area < 0 or np.random.random() < np.exp(-delta_area / T):
            current_x = x_new
            current_area = area_new
            
            if current_area < best_area:
                best_area = current_area
                best_x = current_x.copy()
                
                # If we found a new best, we can reduce step size slightly to refine
                # but keep T to allow further exploration
                # step_size *= 0.95 
        else:
            # Rejected move
            pass
            
        # Cool down
        T *= cooling_rate
        iterations += 1
        
        # If we haven't improved in a long time, we might be stuck in a local minimum.
        # We can try a "kick": a larger perturbation.
        if iterations % 1000 == 0:
             # Check time again
             if time.time() > deadline - 10.0:
                 break
             # Kick: perturb with larger amplitude
             kick_noise = np.random.randn(n) * step_size * 5.0
             x_kick = current_x + kick_noise
             area_kick = _union_area_fast(x_kick)
             if area_kick < current_area:
                 current_x = x_kick
                 current_area = area_kick
                 if current_area < best_area:
                     best_area = current_area
                     best_x = current_x.copy()
    
    # 3. Local Polish with L-BFGS-B on the best SA result
    # SA finds a good basin. L-BFGS-B can polish the continuous part within that basin.
    # We need to re-import minimize or use a simple gradient descent if we want to avoid import overhead,
    # but scipy is available.
    
    try:
        from scipy.optimize import minimize
        
        def _objective_and_grad(x):
            # We need the gradient for L-BFGS-B.
            # We can compute it via finite differences or the analytic subgradient.
            # The analytic subgradient is complex due to sorting.
            # Let's use the analytic subgradient from the original code, adapted for speed.
            
            m = np.arange(M)
            y = (m + 0.5) / M
            i_idx = np.arange(n)
            a = (i_idx + 1) / n
            w = (1.0 - y) / n
            
            L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
            sort_idx = np.argsort(L, axis=0)
            L_sorted = np.take_along_axis(L, sort_idx, axis=0)
            
            d = L_sorted[1:, :] - L_sorted[:-1, :]
            min_d_w = np.minimum(d, w[np.newaxis, :])
            union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
            area = np.mean(union_lengths)
            
            grad = np.zeros(n, dtype=np.float64)
            mask = d < w[np.newaxis, :]
            j_indices, m_indices = np.nonzero(mask)
            
            if len(j_indices) > 0:
                upper_idx = sort_idx[j_indices + 1, m_indices]
                lower_idx = sort_idx[j_indices, m_indices]
                
                weight = 1.0 / M
                grad += np.bincount(upper_idx, weights=np.full_like(upper_idx, weight, dtype=np.float64), minlength=n)
                grad -= np.bincount(lower_idx, weights=np.full_like(lower_idx, weight, dtype=np.float64), minlength=n)
                
            return area, grad

        if time.time() < deadline - 5.0:
            res = minimize(
                _objective_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
    except Exception:
        pass

    # Validation
    if not np.all(np.isfinite(best_x)):
        return x0_keich.astype(np.float64)
    if best_area <= 0:
        return x0_keich.astype(np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
