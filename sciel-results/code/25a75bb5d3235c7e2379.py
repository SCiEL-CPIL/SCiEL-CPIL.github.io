# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _union_area_and_grad(x, n=64, M=5000):
    """
    Compute the union area and its gradient w.r.t. x using M quadrature points.
    Uses a soft-min approximation for differentiability.
    """
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(M) + 0.5) / M
    w_vals = (1.0 - y_vals) / n  # shape (M,)
    
    # L[i, j] = x[j] + a[j] * y[i]  shape (M, n)
    L = x[None, :] + a[None, :] * y_vals[:, None]
    
    # For each row, sort L
    L_sorted = np.sort(L, axis=1)  # (M, n)
    gaps = np.diff(L_sorted, axis=1)  # (M, n-1)
    
    # The union length at height y is w + sum(min(gap_j, w)) for j=0..n-2
    # This is a piecewise linear function. For gradient, we need to know which gaps are < w.
    
    # Compute the union length
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)  # (M,)
    area = np.mean(union_lens)
    
    # Gradient computation:
    # d(area)/d(x_k) = (1/M) * sum_i d(union_len_i)/d(x_k)
    # d(union_len_i)/d(x_k) = sum_j d(min(gap_j, w_i))/d(x_k)
    # 
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # d(gap_j)/d(x_k) = 1 if k is the index of L_sorted[j+1] in original, -1 if k is index of L_sorted[j]
    # 
    # d(min(gap_j, w))/d(x_k) = d(gap_j)/d(x_k) if gap_j < w, else 0
    
    # We need to track which original index corresponds to each sorted position
    sort_idx = np.argsort(L, axis=1)  # (M, n)
    
    # For each height i, for each gap j (between sorted positions j and j+1):
    # If gap_j < w_i, then:
    #   d(gap_j)/d(x_k) = 1 if sort_idx[i, j+1] == k, -1 if sort_idx[i, j] == k, 0 otherwise
    # So the contribution to grad[k] is:
    #   +1 if sort_idx[i, j+1] == k and gap_j < w_i
    #   -1 if sort_idx[i, j] == k and gap_j < w_i
    
    # Efficient computation using bincount
    grad = np.zeros(n)
    
    # For each height, for each gap that is < w:
    #   add +1 to grad[sort_idx[i, j+1]]
    #   add -1 to grad[sort_idx[i, j]]
    
    # Create masks
    mask = gaps < w_vals[:, None]  # (M, n-1)
    
    # For each height i, the indices j where mask[i,j] is True
    # We need to accumulate contributions
    
    # Vectorized approach:
    # For each (i, j) where mask[i,j] is True:
    #   grad[sort_idx[i, j+1]] += 1
    #   grad[sort_idx[i, j]] -= 1
    
    # Flatten and use bincount
    i_idx, j_idx = np.where(mask)
    
    if len(i_idx) > 0:
        # +1 contributions: sort_idx[i_idx, j_idx + 1]
        plus_indices = sort_idx[i_idx, j_idx + 1]
        # -1 contributions: sort_idx[i_idx, j_idx]
        minus_indices = sort_idx[i_idx, j_idx]
        
        # Use bincount
        grad += np.bincount(plus_indices, minlength=n)
        grad -= np.bincount(minus_indices, minlength=n)
    
    grad /= M
    return area, grad

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

def _softmin_objective(x, n=64, M=4000, beta=100.0):
    """
    Smooth approximation of the union area using soft-min.
    The union length at height y is w + sum_j min(gap_j, w).
    min(a, b) ≈ -(1/beta) * log(exp(-beta*a) + exp(-beta*b)) ... no that's not right.
    min(a, b) = a + min(0, b-a) = b + min(0, a-b)
    min(0, z) ≈ -(1/beta) * log(1 + exp(beta*z))  (softplus-based)
    Actually: min(0, z) = -max(0, -z) = -softplus(-z) ≈ -(1/beta)*log(1+exp(-beta*z)) ... 
    Let me use: min(a,b) = 0.5*(a+b - |a-b|) and approximate |.| with soft absolute.
    Or simpler: min(a,b) ≈ a - max(0, a-b) = a - softplus(a-b)/beta approximately... 
    
    Let me just use: min(a, b) ≈ -(1/beta) * log(exp(-beta*a) + exp(-beta*b)) is wrong.
    Correct: min(a,b) = -max(-a, -b) and max(u,v) ≈ (1/beta)*log(exp(beta*u) + exp(beta*v))
    So min(a,b) ≈ -(1/beta)*log(exp(-beta*a) + exp(-beta*b))
    """
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(M) + 0.5) / M
    w_vals = (1.0 - y_vals) / n
    
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)  # (M, n-1)
    
    # min(gap, w) for each element
    # min(gap, w) ≈ -(1/beta) * log(exp(-beta*gap) + exp(-beta*w))
    # But for numerical stability, use: min(gap,w) = gap + min(0, w-gap)
    # min(0, z) ≈ -(1/beta)*log(1 + exp(beta*z)) for z < 0, and 0 for z >> 0
    # Actually: min(0, z) = -max(0, -z) = -softplus(-z)
    # softplus(t) = log(1 + exp(t))
    # So min(0, z) = -log(1 + exp(-z)) ... wait:
    # max(0, -z) = softplus(-z) = log(1 + exp(-z))
    # So min(0, z) = -log(1 + exp(-z))
    
    # Hmm, that's not quite right either. Let me be careful:
    # min(0, z): if z <= 0, result is z; if z > 0, result is 0.
    # softplus(t) = log(1+exp(t)) approximates max(0, t)
    # So -softplus(-z) = -log(1+exp(-z)) approximates min(0, z)
    
    # For numerical stability with large beta:
    z = w_vals[:, None] - gaps  # (M, n-1)
    # min(0, z) ≈ -log(1 + exp(-z)) but this is only the soft version
    # With beta scaling: min(0, z) ≈ -(1/beta)*log(1 + exp(-beta*z))
    
    # For stability:
    # If beta*z > 50: exp(-beta*z) ≈ 0, so -(1/beta)*log(1) = 0 (correct, z>0)
    # If beta*z < -50: exp(-beta*z) ≈ exp(large), log(1+exp(large)) ≈ large, so -(1/beta)*(-beta*z) = z (correct)
    
    bz = beta * z
    # log(1 + exp(-bz)) computed stably
    # log(1 + exp(t)) where t = -bz
    t = -bz
    # For t > 0: log(1+exp(t)) ≈ t + log(1+exp(-t)) ≈ t
    # For t < 0: log(1+exp(t)) ≈ log(1+exp(t)) directly
    log_term = np.where(t > 0, t + np.log1p(np.exp(-t)), np.log1p(np.exp(t)))
    min_0_z = -log_term / beta
    
    union_lens = w_vals + np.sum(min_0_z, axis=1)
    return np.mean(union_lens)

def _softmin_gradient(x, n=64, M=4000, beta=100.0):
    """Gradient of the softmin objective."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(M) + 0.5) / M
    w_vals = (1.0 - y_vals) / n
    
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    sort_idx = np.argsort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    
    z = w_vals[:, None] - gaps  # (M, n-1)
    bz = beta * z
    t = -bz
    
    # d(min(0,z))/dz = -1/(1+exp(beta*z)) = -sigmoid(beta*z) ... let me derive
    # min(0,z) ≈ -(1/beta)*log(1+exp(-beta*z))
    # d/dz = -(1/beta) * (-beta*exp(-beta*z))/(1+exp(-beta*z)) = exp(-beta*z)/(1+exp(-beta*z)) = 1/(1+exp(beta*z))
    # = sigmoid(-beta*z)
    
    # For stability: sigmoid(-bz) = 1/(1+exp(bz))
    # If bz > 0: exp(bz) large, sigmoid ≈ exp(-bz)
    # If bz < 0: exp(bz) small, sigmoid ≈ 1
    
    sig = np.where(bz > 0, np.exp(-bz) / (1 + np.exp(-bz)), 1.0 / (1.0 + np.exp(bz)))
    
    # d(union_len)/d(gap_j) = sig[i,j] for each (i,j)
    # d(gap_j)/d(x_k): same as before
    # d(gap_j) = d(L_sorted[j+1] - L_sorted[j])
    # = 1 if x_k is at sorted position j+1, -1 if at sorted position j
    
    grad = np.zeros(n)
    
    # For each (i, j): contribution is sig[i,j] * (indicator_k_at_pos_j+1 - indicator_k_at_pos_j)
    # Vectorized:
    plus_indices = sort_idx[:, 1:].reshape(-1)  # (M*(n-1),)
    minus_indices = sort_idx[:, :-1].reshape(-1)  # (M*(n-1),)
    sig_flat = sig.reshape(-1)
    
    grad += np.bincount(plus_indices, weights=sig_flat, minlength=n)
    grad -= np.bincount(minus_indices, weights=sig_flat, minlength=n)
    
    grad /= M
    return grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 615  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    np.random.seed(42)
    
    # Phase 1: Soft-min optimization with increasing beta
    betas = [20, 50, 100, 200, 400]
    for beta in betas:
        if time.time() > deadline - 120:
            break
        
        def obj_soft(x):
            return _softmin_objective(x, n=n, M=4000, beta=beta)
        
        def grad_soft(x):
            return _softmin_gradient(x, n=n, M=4000, beta=beta)
        
        try:
            result = minimize(
                obj_soft, best_x, method='L-BFGS-B',
                jac=grad_soft,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun < best_area:
                # Verify with exact area
                exact_area = _compute_area(result.x)
                if exact_area < best_area:
                    best_area = exact_area
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement on exact objective
    def objective_exact(x):
        a = (np.arange(n) + 1) / n
        y_vals = (np.arange(6000) + 0.5) / 6000
        w_vals = (1.0 - y_vals) / n
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
        return np.mean(union_lens)
    
    if time.time() < deadline - 100:
        result = minimize(
            objective_exact, best_x, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Phase 3: Multiple Nelder-Mead restarts with perturbations
    for restart in range(6):
        if time.time() > deadline - 60:
            break
        scale = 0.003 * (1 + restart * 0.3)
        x_start = best_x + np.random.normal(0, scale, size=n)
        result = minimize(
            objective_exact, x_start, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Phase 4: Final subgradient polish
    if time.time() < deadline - 30:
        # Use exact gradient for a few steps
        step_size = 0.001
        for step in range(200):
            if time.time() > deadline - 10:
                break
            area, grad = _union_area_and_grad(best_x, n=n, M=5000)
            if area < best_area:
                best_area = area
            
            # Adaptive step size
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-10:
                break
            
            # Simple adaptive: reduce step if area increases
            x_trial = best_x - step_size * grad / norm_grad * norm_grad
            area_trial = objective_exact(x_trial)
            if area_trial < area:
                best_x = x_trial
                step_size = min(step_size * 1.1, 0.01)
            else:
                step_size *= 0.7
                if step_size < 1e-7:
                    break
    
    # Verify with full precision
    final_area = _compute_area(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
