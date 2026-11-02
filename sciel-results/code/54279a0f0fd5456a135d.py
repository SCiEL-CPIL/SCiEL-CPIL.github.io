"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps_fft(h):
    """Compute R_k for k=1..N-1 using FFT. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2*n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    # Autocorrelation
    corr = np.fft.irfft(H * np.conj(H), size)[:n]
    # corr[k] = sum_i h[i] * h[i+k] for k=0..n-1
    # We want R_k = (2/N) * corr[k] for k=1..n-1
    return (2.0 / n) * corr[1:]

def _compute_overlaps_direct(h):
    """Compute R_k for k=1..N-1 directly. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1."""
    n = len(h)
    if n <= 256:
        return _compute_overlaps_fft(h)
    else:
        return _compute_overlaps_direct(h)

def _soft_min_obj_and_grad(params, beta):
    """
    Objective and gradient for symmetric h.
    params: length N/2, represents h[0:N/2]
    h is constructed symmetrically.
    """
    n_half = len(params)
    n = 2 * n_half
    # Construct full h
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    
    corrs = _compute_overlaps(h)
    
    # Soft min of corrs
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient
    # d(soft_min)/d(corrs_j) = exp(-beta * corrs_j) / sum(exp(-beta * corrs))
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg
    
    # d(soft_min)/d(h) = sum_j weights_j * d(corrs_j)/d(h)
    # Use FFT-based gradient for efficiency
    grad_h = np.zeros(n)
    n = len(h)
    size = 1
    while size < 2*n - 1:
        size *= 2
    
    # For each lag k, the contribution to grad_h is:
    # d(corrs_k)/d(h[i]) = h[i+k] for i < n-k, and h[i-k] for i >= k
    # So grad_h[i] = sum_k weights[k] * (2/n) * (h[i+k] if i+k < n else 0) 
    #                + sum_k weights[k] * (2/n) * (h[i-k] if i-k >= 0 else 0)
    
    # This is a convolution-like operation. Let's compute it directly but efficiently.
    # Actually, let's use the direct method since n=100 is small
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w == 0: continue
        grad_h[:n-k] += w * h[k:]
        grad_h[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = - grad_h
    grad_h = -grad_h
    
    # Map grad_h back to params (symmetric)
    # h[i] = params[i] for i < n/2
    # h[n-1-i] = params[i]
    # d(obj)/d(params[i]) = grad_h[i] + grad_h[n-1-i]
    grad_params = grad_h[:n_half] + grad_h[n-1:n_half-1:-1]
    
    return obj, grad_params

def _project_symmetric(params, target_param_sum):
    """Project symmetric params to satisfy sum(params) = target_param_sum and 0<=params<=1."""
    n_half = len(params)
    
    p = params.copy()
    p = np.clip(p, 0, 1)
    diff = target_param_sum - np.sum(p)
    if abs(diff) > 1e-12:
        for _ in range(10):
            diff = target_param_sum - np.sum(p)
            if abs(diff) < 1e-12: break
            mask = (p > 0) & (p < 1)
            if np.sum(mask) == 0:
                break
            p[mask] += diff / np.sum(mask)
            p = np.clip(p, 0, 1)
    return p

def _fista_minimize(params, target_param_sum, max_iter=5000, tol=1e-10):
    """
    FISTA for minimizing 1 - min(corrs) subject to 0<=h<=1, sum(h)=n/2.
    Uses symmetric parameterization.
    """
    n_half = len(params)
    n = 2 * n_half
    
    x = _project_symmetric(params, target_param_sum)
    y = x.copy()
    t = 1.0
    
    # Lipschitz constant estimate for the gradient of the smooth part
    # The gradient of soft_min has bounded norm. Let's estimate it.
    L = 1.0
    for _ in range(10):
        # Estimate L by checking gradient change
        x_test = x + 0.1 * np.random.randn(n_half)
        x_test = np.clip(x_test, 0, 1)
        obj1, g1 = _soft_min_obj_and_grad(x, 100.0)
        obj2, g2 = _soft_min_obj_and_grad(x_test, 100.0)
        diff_g = np.linalg.norm(g1 - g2)
        diff_x = np.linalg.norm(x - x_test)
        if diff_x > 1e-10:
            L = max(L, diff_g / diff_x)
        x = _project_symmetric(x, target_param_sum)
    
    L = max(L, 1.0)
    step = 1.0 / L
    
    for it in range(max_iter):
        if _time_remaining() < 3.0:
            break
        
        # Gradient step
        obj_y, grad_y = _soft_min_obj_and_grad(y, 100.0)
        x_new = y - step * grad_y
        
        # Proximal step: project onto {0<=x<=1, sum(x)=target_param_sum}
        x_new = _project_symmetric(x_new, target_param_sum)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
        
        # FISTA update
        t_new = (1 + np.sqrt(1 + 4*t*t)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        y = _project_symmetric(y, target_param_sum)
        x = x_new
        t = t_new
    
    return x

def _local_search(h, target_sum, max_iter=10000):
    """
    Local search: try flipping pairs of indices to improve the min overlap.
    """
    n = len(h)
    h = h.copy()
    h = np.clip(h, 0, 1)
    
    # Ensure sum constraint
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    best_min = np.min(_compute_overlaps(h))
    
    for it in range(max_iter):
        if _time_remaining() < 3.0:
            break
        
        improved = False
        # Try random pairs of indices
        for _ in range(100):
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            # Try moving mass from i to j or vice versa
            for direction in [1, -1]:
                for delta in [0.01, 0.05, 0.1]:
                    h_test = h.copy()
                    h_test[i] += direction * delta
                    h_test[j] -= direction * delta
                    
                    if h_test[i] < 0 or h_test[i] > 1 or h_test[j] < 0 or h_test[j] > 1:
                        continue
                    
                    # Check sum constraint (it's preserved)
                    corrs = _compute_overlaps(h_test)
                    min_val = np.min(corrs)
                    
                    if min_val > best_min + 1e-10:
                        h = h_test
                        best_min = min_val
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return h

def construct():
    n = N
    n_half = n // 2
    target_sum = n / 2.0
    target_param_sum = n / 4.0
    
    best_params = np.full(n_half, 0.5)
    best_obj = float('inf')
    
    # Calculate initial best
    h_init = np.concatenate([best_params, best_params[::-1]])
    corrs_init = _compute_overlaps(h_init)
    best_obj = 1.0 - np.min(corrs_init)
    
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n_half, 0.5))
    
    # Symmetric gradients
    restarts.append(np.linspace(0, 1, n_half))
    restarts.append(np.linspace(1, 0, n_half))
    
    # Symmetric sinusoids
    x = np.linspace(0, 2*np.pi, n_half)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.25*np.sin(3*x))
    restarts.append(0.5 + 0.25*np.cos(3*x))
    
    # Random symmetric
    for _ in range(30):
        restarts.append(np.random.uniform(0, 1, n_half))
    
    # Block patterns
    for k in range(1, n_half):
        restarts.append(np.concatenate([np.zeros(k), np.ones(n_half-k)*0.5]))
    
    # Optimize each restart
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        params = _project_symmetric(x0, target_param_sum)
        
        # Multi-stage beta optimization with L-BFGS-B
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if _time_remaining() < 2.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                params_new = _project_symmetric(res.x, target_param_sum)
                h_new = np.concatenate([params_new, params_new[::-1]])
                corrs_new = _compute_overlaps(h_new)
                obj_new = 1.0 - np.min(corrs_new)
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_params = params_new
                    params = params_new
            except Exception:
                pass
        
        # FISTA refinement
        if _time_remaining() > 5.0:
            try:
                params_fista = _fista_minimize(params, target_param_sum, max_iter=3000)
                h_fista = np.concatenate([params_fista, params_fista[::-1]])
                corrs_fista = _compute_overlaps(h_fista)
                obj_fista = 1.0 - np.min(corrs_fista)
                if obj_fista < best_obj:
                    best_obj = obj_fista
                    best_params = params_fista
            except Exception:
                pass
    
    # Final polish with full vector optimization
    if _time_remaining() > 10.0:
        h_full = np.concatenate([best_params, best_params[::-1]])
        
        def full_obj_and_grad(h):
            corrs = _compute_overlaps(h)
            min_val = np.min(corrs)
            shifted = corrs - min_val
            log_sum = np.log(np.sum(np.exp(-500.0 * shifted)))
            soft_min = min_val - log_sum / 500.0
            obj = 1.0 - soft_min
            
            exp_neg_beta = np.exp(-500.0 * corrs)
            sum_exp_neg = np.sum(exp_neg_beta)
            weights = exp_neg_beta / sum_exp_neg
            
            grad_h = np.zeros(n)
            for k in range(1, n):
                w = weights[k-1] * (2.0 / n)
                if w == 0: continue
                grad_h[:n-k] += w * h[k:]
                grad_h[k:] += w * h[:n-k]
            
            return obj, -grad_h

        try:
            res = minimize(
                full_obj_and_grad,
                h_full,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            h_new = np.clip(res.x, 0, 1)
            diff = target_sum - np.sum(h_new)
            if abs(diff) > 1e-12:
                mask = (h_new > 0) & (h_new < 1)
                if np.sum(mask) > 0:
                    h_new[mask] += diff / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
            
            corrs_new = _compute_overlaps(h_new)
            obj_new = 1.0 - np.min(corrs_new)
            if obj_new < best_obj:
                best_obj = obj_new
                final_h = h_new
            else:
                final_h = np.concatenate([best_params, best_params[::-1]])
        except Exception:
            final_h = np.concatenate([best_params, best_params[::-1]])
    else:
        final_h = np.concatenate([best_params, best_params[::-1]])
    
    # Local search polish
    if _time_remaining() > 5.0:
        try:
            h_ls = _local_search(final_h, target_sum, max_iter=5000)
            corrs_ls = _compute_overlaps(h_ls)
            obj_ls = 1.0 - np.min(corrs_ls)
            if obj_ls < best_obj:
                best_obj = obj_ls
                final_h = h_ls
        except Exception:
            pass
    
    # Final adjustment for exact sum
    diff = target_sum - np.sum(final_h)
    if abs(diff) > 1e-12:
        mask = (final_h > 0) & (final_h < 1)
        if np.sum(mask) > 0:
            final_h[mask] += diff / np.sum(mask)
            final_h = np.clip(final_h, 0, 1)
    
    return final_h
# EVOLVE-BLOCK-END
