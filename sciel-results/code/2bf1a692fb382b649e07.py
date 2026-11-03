"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 138.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx.
    
    We evaluate at s = 2k/n for k = 1, ..., n-1 (shifts within (0,2)).
    R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    We want to maximize min_k R(2k/n).
    """
    n = len(h)
    # Use FFT to compute all correlations efficiently
    # For linear (non-circular) correlation:
    # corr[k] = sum_{i=0}^{n-k-1} h[i] * h[i+k] for k=1,...,n-1
    # We can compute this using FFT by zero-padding
    n_fft = 2 * n
    H = np.fft.fft(h, n=n_fft)
    # For autocorrelation via FFT with zero-padding:
    # ifft(|H|^2)[k] = sum_i h[i] * h[i-k] (circular, but with zero-padding it's linear)
    # Actually for linear autocorrelation:
    # r[k] = sum_{i=0}^{n-1-k} h[i] * h[i+k]
    # This equals ifft(fft(h, 2n) * conj(fft(h, 2n)))[k] for k=0..n-1
    # But we need to be careful about the indexing
    R = np.fft.ifft(H * np.conj(H)).real
    # R[k] for k=0..n-1 gives sum_i h[i]*h[i-k mod 2n], which with zero padding
    # becomes the linear autocorrelation
    # R[0] = sum h[i]^2
    # R[k] = sum_{i=k}^{n-1} h[i]*h[i-k] = sum_{i=0}^{n-1-k} h[i+k]*h[i] for k>0
    # So R[k] = corr[k] for k >= 0
    corrs = R[1:n] * (2.0 / n)
    return np.min(corrs)

def _get_all_corrs(h):
    """Get all cross-correlations R(2k/n) for k=1,...,n-1 using FFT."""
    n = len(h)
    n_fft = 2 * n
    H = np.fft.fft(h, n=n_fft)
    R = np.fft.ifft(H * np.conj(H)).real
    corrs = R[1:n] * (2.0 / n)
    return corrs

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp."""
    corrs = _get_all_corrs(h)
    
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_soft(x, beta=100.0):
    """Objective: minimize 1 - soft_min_corr, approximating max overlap."""
    h = x
    soft_min, _ = _soft_min_cross_correlation(h, beta)
    obj = 1.0 - soft_min
    return obj

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation with FFT-based gradient."""
    h = x
    n = len(h)
    
    # Compute all correlations using FFT
    corrs = _get_all_corrs(h)
    
    # Soft min using cleaner formulation:
    # soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg  # softmax weights
    
    soft_min = min_val - np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    # Gradient: d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(corrs_k)/d(h[j]) = (2/n) * [h[j+k] if j < n-k] + (2/n) * [h[j-k] if j >= k]
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h) = -grad
    grad_obj = -grad
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # First project onto {x >= 0, sum = target_sum} using bisection
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
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
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # More alternating patterns
    for period in [3, 4, 5, 6, 7]:
        pattern = [0.0 if (i % period) < period//2 else 1.0 for i in range(n)]
        mean_val = np.mean(pattern)
        if mean_val > 0:
            pattern = [p / (2*mean_val) for p in pattern]
        restarts.append(np.array(pattern))
    # Random
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal with various frequencies
    x_grid = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5, 6, 8, 10]:
        restarts.append(0.5 + 0.3*np.sin(freq*x_grid))
        restarts.append(0.5 + 0.3*np.cos(freq*x_grid))
    # Two-tone patterns
    for ratio in [0.2, 0.3, 0.4, 0.5, 0.6]:
        k = int(n * ratio)
        if 0 < k < n:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 1.0/(1-ratio))]))
    
    # Optimize each restart with soft-min objective
    for x0 in restarts:
        if _time_remaining() < 3.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        # Use soft-min with moderate beta for smoothness
        try:
            res = minimize(
                _objective_and_grad_soft,
                x,
                args=(50.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = 1.0 - _compute_cross_correlation_min(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 8.0:
        for beta in [100.0, 200.0, 500.0, 1000.0, 2000.0]:
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
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Final polish with hard min using subgradient
    if _time_remaining() > 5.0:
        step_size = 0.02
        for iteration in range(200):
            if _time_remaining() < 3.0:
                break
            h = best_x
            
            # Find which k achieves the minimum using FFT
            corrs = _get_all_corrs(h)
            k_star = np.argmin(corrs) + 1  # k_star is 1-indexed
            
            # Subgradient: d(min_corr)/d(h) at the minimizing k
            grad = np.zeros(n)
            k = k_star
            w = (2.0 / n)
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
            
            # Take a step in the +grad direction to increase min_corr
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
                step_size = min(step_size * 1.05, 0.15)
            else:
                step_size *= 0.7
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-14:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
