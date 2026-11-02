"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1
    using FFT for O(n log n) speed."""
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    corrs = acf[1:n] * (2.0 / n)
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp."""
    corrs = _compute_all_corrs_fft(h)
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation with FFT-based correlations."""
    h = x
    n = len(h)
    
    # Compute all correlations via FFT
    corrs = _compute_all_corrs_fft(h)
    
    # Clean soft-min formulation: soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = min_val - np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    weights = exp_neg_beta / sum_exp_neg
    w_k = weights * (2.0 / n)
    
    # Efficient gradient: d(soft_min)/d(h[j]) = sum_{k=1}^{n-1} w_k * (h[j+k] + h[j-k])
    # Use FFT-based convolution for speed
    if n > 1:
        # For each k, the contribution to grad[j] from w_k is:
        #   w_k * h[j+k] (for j < n-k) + w_k * h[j-k] (for j >= k)
        # This is equivalent to convolving h with a symmetric kernel
        # Build kernel: for shift k, we add w_k to positions
        # grad[j] = sum_k w_k * (h[j+k] if j+k<n else 0) + sum_k w_k * (h[j-k] if j-k>=0 else 0)
        # = sum_k w_k * h[j+k] (truncated) + sum_k w_k * h[j-k] (truncated)
        # This is the linear cross-correlation of h with itself weighted by w_k
        
        # Use FFT: grad = ifft(fft(h) * conj(fft(w_padded))) appropriately
        # Actually, let's use a direct but vectorized approach with rolling
        grad = np.zeros(n)
        for k in range(1, n):
            wk = w_k[k-1]
            if wk < 1e-15:
                continue
            # h[j+k] contributes to grad[j] for j=0..n-k-1
            grad[:n-k] += wk * h[k:]
            # h[j-k] contributes to grad[j] for j=k..n-1
            grad[k:] += wk * h[:n-k]
        grad_obj = -grad
    else:
        grad_obj = np.zeros(n)
        
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum} using bisection on water level."""
    n = len(v)
    
    # Bisection on the water level theta such that sum(max(0, min(1, v - theta))) = target_sum
    lo = np.min(v) - 1.0
    hi = np.max(v) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        w = np.clip(v - mid, 0.0, 1.0)
        s = np.sum(w)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    
    theta = (lo + hi) / 2.0
    w = np.clip(v - theta, 0.0, 1.0)
    
    # Fine-tune to hit exact sum
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-14:
        # Distribute among non-saturated elements
        mask = (w > 1e-10) & (w < 1.0 - 1e-10)
        cnt = np.sum(mask)
        if cnt > 0:
            w[mask] += diff / cnt
            w = np.clip(w, 0.0, 1.0)
        else:
            # All saturated or zero; adjust one element
            if diff > 0:
                mask = w < 1.0
            else:
                mask = w > 0.0
            cnt = np.sum(mask)
            if cnt > 0:
                w[mask] += diff / cnt
                w = np.clip(w, 0.0, 1.0)
    
    return w

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        if k > 0 and n - k > 0:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # 3-level patterns
    restarts.append(np.array([0.0 if i % 3 == 0 else (2.0/3.0 if i % 3 == 1 else 1.0) for i in range(n)]))
    restarts.append(np.array([1.0 if i % 3 == 0 else (0.0 if i % 3 == 1 else 2.0/3.0) for i in range(n)]))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.35*np.sin(3*x + np.pi/4))
    # Smooth random (low-pass filtered)
    for _ in range(5):
        raw = np.random.uniform(0, 1, n)
        # Simple moving average smoothing
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(raw, kernel, mode='same')
        smooth = np.clip(smooth, 0, 1)
        restarts.append(smooth)
    
    # Phase 1: Soft-min optimization with increasing beta per restart
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: High-beta polish on best solution
    if _time_remaining() > 12.0:
        for beta in [500.0, 1000.0, 2000.0, 5000.0]:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 3: Local search with subgradient on the active constraint (FFT-accelerated)
    if _time_remaining() > 8.0:
        step_size = 0.01
        for _ in range(300):
            if _time_remaining() < 3.0:
                break
            h = best_x
            n = len(h)
            
            # Use FFT to find min correlation and compute subgradient
            corrs = _compute_all_corrs_fft(h)
            k_star = np.argmin(corrs) + 1  # +1 because corrs[0] corresponds to k=1
            
            grad = np.zeros(n)
            wk = 2.0 / n
            # h[j+k] contributes to grad[j] for j=0..n-k-1
            grad[:n-k_star] += wk * h[k_star:]
            # h[j-k] contributes to grad[j] for j=k..n-1
            grad[k_star:] += wk * h[:n-k_star]
            
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
            else:
                step_size *= 0.5
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 1e-10) & (best_x < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
        else:
            mask = best_x < 1.0 if diff > 0 else best_x > 0.0
            cnt = np.sum(mask)
            if cnt > 0:
                best_x[mask] += diff / cnt
                best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
