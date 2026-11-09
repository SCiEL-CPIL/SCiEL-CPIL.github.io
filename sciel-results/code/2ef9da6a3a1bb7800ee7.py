# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area_and_grad(x, M=4000):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)  # shape (M, n)
    L_sorted = np.take_along_axis(L, order, axis=1)  # shape (M, n)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    
    # mask: (M, n-1) boolean, True where diffs < w
    mask = diffs < w[:, np.newaxis]  # shape (M, n-1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient computation using bincount for speed
    # For each active pair (m, j): +1 to order[m, j+1], -1 to order[m, j]
    m_idx, j_idx = np.where(mask)
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        # Use bincount for efficient accumulation
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        grad = grad_right - grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M  # mean over quadrature points
    
    return area, grad

def _compute_area_exact(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x (full precision)."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0

def _softmin_objective(x, beta=100.0, M=4000):
    """
    Smooth surrogate for the union area using soft-min approximation.
    This avoids the nonsmoothness of the min function in the union length calculation.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    L_sorted = np.sort(L, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    # Soft-min approximation: min(d, w) ≈ -log(sum(exp(-beta*d), exp(-beta*w))) / beta
    # But we need min(diffs, w), so:
    # soft_min = -1/beta * log(exp(-beta*diffs) + exp(-beta*w))
    # For numerical stability:
    # soft_min = min_val - 1/beta * log(1 + exp(-beta*(max_val - min_val)))
    # where min_val = min(diffs, w) and max_val = max(diffs, w)
    
    # Actually, let's use a simpler approach:
    # min(d, w) = (d + w - |d - w|) / 2
    # And |d - w| ≈ (1/beta) * log(exp(beta*(d-w)) + exp(beta*(w-d)))
    # But this is complex. Let's use the log-sum-exp trick directly.
    
    # For each pair (m, j), we have diff = diffs[m,j] and width = w[m]
    # min(diff, width) can be approximated by:
    #   -1/beta * log(exp(-beta*diff) + exp(-beta*width))
    
    # To avoid overflow, use the log-sum-exp trick:
    #   log(exp(-beta*diff) + exp(-beta*width)) = 
    #     max(-beta*diff, -beta*width) + log(exp(-beta*diff - max) + exp(-beta*width - max))
    
    neg_beta_diffs = -beta * diffs
    neg_beta_w = -beta * w[:, np.newaxis]
    
    max_vals = np.maximum(neg_beta_diffs, neg_beta_w)
    log_sum_exp = max_vals + np.log(np.exp(neg_beta_diffs - max_vals) + np.exp(neg_beta_w - max_vals))
    
    soft_min = -log_sum_exp / beta
    
    union_len = w + np.sum(soft_min, axis=1)
    return np.mean(union_len)

def _softmin_gradient(x, beta=100.0, M=4000):
    """
    Compute gradient of the soft-min surrogate objective.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    neg_beta_diffs = -beta * diffs
    neg_beta_w = -beta * w[:, np.newaxis]
    
    max_vals = np.maximum(neg_beta_diffs, neg_beta_w)
    exp_diffs = np.exp(neg_beta_diffs - max_vals)
    exp_w = np.exp(neg_beta_w - max_vals)
    log_sum_exp = max_vals + np.log(exp_diffs + exp_w)
    
    # Gradient of soft_min w.r.t. diff:
    # d(soft_min)/d(diff) = -d(-log_sum_exp/beta)/d(diff)
    # = (1/beta) * d(log_sum_exp)/d(diff)
    # d(log_sum_exp)/d(diff) = -beta * exp_diffs / (exp_diffs + exp_w)
    # So d(soft_min)/d(diff) = -exp_diffs / (exp_diffs + exp_w)
    
    # And d(diff)/d(L_sorted[j+1]) = 1, d(diff)/d(L_sorted[j]) = -1
    # So d(soft_min)/d(L_sorted[j+1]) = -exp_diffs / (exp_diffs + exp_w)
    # d(soft_min)/d(L_sorted[j]) = exp_diffs / (exp_diffs + exp_w)
    
    weight_right = -exp_diffs / (exp_diffs + exp_w)  # d(soft_min)/d(L_sorted[j+1])
    weight_left = exp_diffs / (exp_diffs + exp_w)   # d(soft_min)/d(L_sorted[j])
    
    # Now accumulate gradients
    # For each (m, j): 
    #   grad[order[m, j+1]] += weight_right[m,j] / M
    #   grad[order[m, j]] += weight_left[m,j] / M
    
    m_idx, j_idx = np.where(np.ones_like(diffs, dtype=bool))  # all pairs
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_left = np.bincount(left_indices, weights=weight_left[m_idx, j_idx], minlength=n)
        grad_right = np.bincount(right_indices, weights=weight_right[m_idx, j_idx], minlength=n)
        grad = grad_right + grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M
    
    return grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 630.0
    safety_margin = 30.0
    
    # Keich construction as initial guess
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area_exact(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: L-BFGS-B with analytical gradient (fast M=4000)
    initial_guesses = [x0.copy()]
    
    # Add perturbed versions
    for attempt in range(8):
        perturbation_scale = 0.3 / (attempt + 1)
        perturbed = x0 + rng.normal(0, perturbation_scale, size=n)
        initial_guesses.append(perturbed)
    
    # Add some random starts
    for attempt in range(4):
        random_start = rng.uniform(-1, 1, size=n)
        initial_guesses.append(random_start)
    
    for idx, x_start in enumerate(initial_guesses):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=4000)[0],
                x_start,
                method='L-BFGS-B',
                jac=lambda x: _union_area_and_grad(x, M=4000)[1],
                options={
                    'maxiter': 150,
                    'maxfun': 400,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate = _compute_area_exact(result.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Strategy 2: Soft-min surrogate with increasing beta
    if time.time() < deadline - safety_margin:
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - safety_margin:
                break
            
            try:
                result = minimize(
                    lambda x: _softmin_objective(x, beta=beta, M=4000),
                    best_x,
                    method='L-BFGS-B',
                    jac=lambda x: _softmin_gradient(x, beta=beta, M=4000),
                    options={
                        'maxiter': 100,
                        'maxfun': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20,
                    }
                )
                
                if np.all(np.isfinite(result.x)):
                    area_candidate = _compute_area_exact(result.x)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = result.x.copy()
                        best_area = area_candidate
            except Exception:
                pass
    
    # Strategy 3: Final polish with full precision gradient
    if time.time() < deadline - safety_margin:
        try:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=20000)[0],
                best_x,
                method='L-BFGS-B',
                jac=lambda x: _union_area_and_grad(x, M=20000)[1],
                options={
                    'maxiter': 50,
                    'maxfun': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate = _compute_area_exact(result.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Strategy 4: Nelder-Mead refinement (non-gradient based, can escape local minima)
    if time.time() < deadline - 60:
        try:
            def objective_fast(x):
                return _union_area_and_grad(x, M=5000)[0]
            
            result = minimize(
                objective_fast, best_x, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate = _compute_area_exact(result.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Strategy 5: More L-BFGS-B restarts from perturbed best
    for restart in range(5):
        if time.time() > deadline - safety_margin:
            break
        
        scale = 0.01 * (1 + restart * 0.5)
        x_start = best_x + rng.normal(0, scale, size=n)
        
        try:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=4000)[0],
                x_start,
                method='L-BFGS-B',
                jac=lambda x: _union_area_and_grad(x, M=4000)[1],
                options={
                    'maxiter': 100,
                    'maxfun': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate = _compute_area_exact(result.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Final verification
    final_area = _compute_area_exact(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
