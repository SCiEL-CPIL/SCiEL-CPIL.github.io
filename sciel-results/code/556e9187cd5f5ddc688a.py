"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # First project onto {x >= 0, sum = target_sum}
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
        if abs(diff) < 1e-13:
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

def _compute_all_corrs_fft(h):
    """Compute R_k = (1/2) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1."""
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    # acf[k] = sum_{i=0}^{n-k} h[i]h[i+k]
    # We want (1/2) * sum_{i=0}^{n-k-1} ...
    # Note: acf[k] includes h[n-k]h[n-k] if we go to n-1?
    # Standard autocorrelation via FFT:
    # Let h_padded be h padded with zeros to 2n.
    # IFFT(|FFT(h_padded)|^2) gives circular autocorrelation of length 2n.
    # The first n values are the linear autocorrelation r[k] = sum_{i=0}^{n-k-1} h[i]h[i+k].
    # So R_k = 0.5 * acf[k].
    corrs = acf[1:n] * 0.5
    return corrs

def _compute_grad_J(h, T):
    """Compute gradient of J(h) = sum_k max(0, R_k - T)^2."""
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    
    # Active constraints: R_k > T
    active = corrs > T
    if not np.any(active):
        return np.zeros(n)
    
    # Gradient of R_k w.r.t h
    # d(R_k)/d(h_j) = 0.5 * (h[j+k] + h[j-k])
    # Sum over active k: 2 * (R_k - T) * 0.5 * (h[j+k] + h[j-k])
    
    grad = np.zeros(n)
    # Weights for each shift k
    # w_k = (R_k - T) for active k, else 0
    weights = np.where(active, corrs - T, 0.0)
    
    # Efficient gradient calculation using FFT?
    # grad[j] = sum_{k=1}^{n-1} weights[k-1] * 0.5 * (h[j+k] + h[j-k])
    # This is a linear convolution/correlation.
    # Let's use the direct loop if n is small, or FFT if large.
    # n=100, loop is fine but FFT is safer for speed.
    
    # Construct signal g of length 2n for correlation
    # g[k] = weights[k-1] for k=1..n-1, g[0]=0, g[n..]=0
    g = np.zeros(2 * n)
    g[1:n] = weights
    
    # h_padded
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    
    # Linear correlation of h and g
    # corr = IFFT(FFT(h_padded) * conj(FFT(g)))
    # This gives sum_i h[i] g[i-k] ? No.
    # Standard: IFFT(A * conj(B)) gives cross-correlation.
    # We want sum_k w_k h[j+k] and sum_k w_k h[j-k].
    
    # Let's just do the loop for n=100, it's very fast.
    # 100 * 100 = 10k ops. Negligible.
    for k in range(1, n):
        w = weights[k-1]
        if w != 0.0:
            # h[j+k] term: j from 0 to n-k-1
            grad[:n-k] += w * h[k:]
            # h[j-k] term: j from k to n-1
            grad[k:] += w * h[:n-k]
            
    return grad

def _minimize_J_for_T(T, h_init, max_iter=1000):
    """Minimize J(h) for fixed T using PGD."""
    n = N
    target_sum = n / 2.0
    h = h_init.copy()
    h = _project_to_simplex(h, target_sum)
    
    best_J = float('inf')
    best_h = h.copy()
    
    alpha = 0.1 # Step size
    for i in range(max_iter):
        if _time_remaining() < 2.0:
            break
            
        corrs = _compute_all_corrs_fft(h)
        J = np.sum(np.maximum(0, corrs - T)**2)
        
        if J < best_J:
            best_J = J
            best_h = h.copy()
            
        if J < 1e-8:
            break
            
        grad = _compute_grad_J(h, T)
        
        # Adaptive step size?
        # For now, fixed with decay
        h_new = h - alpha * grad
        h = _project_to_simplex(h_new, target_sum)
        
        if i % 100 == 0 and i > 0:
            alpha *= 0.95
            
    return best_J, best_h

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    # Initial guess
    np.random.seed(42)
    h_init = np.random.uniform(0, 1, n)
    h_init = _project_to_simplex(h_init, target_sum)
    
    # Binary search for T
    # Lower bound: Theoretical min for N=100 is around 0.25?
    # For random, max corr is higher.
    # Let's start wide.
    T_low = 0.15
    T_high = 0.5
    
    best_h = h_init.copy()
    best_T = T_high
    
    # First, find a feasible T_high
    # Start with uniform 0.5
    h_uni = np.full(n, 0.5)
    corrs_uni = _compute_all_corrs_fft(h_uni)
    max_corr_uni = np.max(corrs_uni)
    T_high = max_corr_uni + 0.05
    
    # Optimize at T_high to get a good starting point
    _, best_h = _minimize_J_for_T(T_high, h_uni, max_iter=500)
    
    # Refine T_high to be the actual max corr of best_h
    corrs_best = _compute_all_corrs_fft(best_h)
    T_high = np.max(corrs_best)
    
    # Binary search
    for _ in range(25):
        if _time_remaining() < 10.0:
            break
            
        T_mid = (T_low + T_high) / 2.0
        
        # Try to find h such that max R_k <= T_mid
        # We minimize J(h) = sum max(0, R_k - T_mid)^2
        # If min J is small, T_mid is feasible.
        min_J, h_candidate = _minimize_J_for_T(T_mid, best_h, max_iter=800)
        
        # Check actual max corr of the candidate
        corrs_cand = _compute_all_corrs_fft(h_candidate)
        actual_max = np.max(corrs_cand)
        
        if actual_max <= T_mid + 1e-5:
            # Feasible
            T_high = T_mid
            best_h = h_candidate.copy()
        else:
            # Infeasible, or we didn't converge well enough
            # If min_J is small but actual_max > T_mid, it might be a local min or tolerance issue.
            # If min_J is large, definitely infeasible.
            T_low = T_mid
            
    # Final polish: Optimize for the best T_high found
    # We can run a few more iterations at T_high
    min_J, best_h = _minimize_J_for_T(T_high, best_h, max_iter=2000)
    
    # Ensure exact sum and bounds
    best_h = np.clip(best_h, 0, 1)
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 0) & (best_h < 1)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
            
    return best_h
# EVOLVE-BLOCK-END
