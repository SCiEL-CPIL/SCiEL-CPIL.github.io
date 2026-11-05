import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _subgradient_step(h, alpha):
    """
    Perform one step of subgradient descent on V(h) assuming sum(h)=1.
    If sum(h) is not 1, we first normalize h to sum 1, compute subgradient,
    take step, and return the new h (which will sum to 1).
    """
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return h
    
    # Normalize to sum 1 for stable subgradient calculation
    h_norm = h / s
    
    conv = np.convolve(h_norm, h_norm)
    max_val = np.max(conv)
    
    # Find indices where max is achieved (within tolerance)
    # Using a small tolerance to capture "plateaus" in the max
    tol = 1e-9 
    max_indices = np.where(np.abs(conv - max_val) < tol)[0]
    
    # Compute subgradient
    # g_j = sum_{k in max_indices} d(conv_k)/d(h_j)
    # d(conv_k)/d(h_j) = h_{k-j} + h_{j-k} (with bounds check)
    # We can compute this efficiently.
    # Let v = 0
    # For each k in max_indices:
    #   v += h[k-j] + h[j-k]
    
    # Vectorized approach:
    # The term h[k-j] is a shifted version of h.
    # If we define a vector w such that w_l = 1 if l in max_indices, else 0.
    # Then the subgradient is essentially the sum of shifted h's.
    # Specifically, grad_j = sum_k w_k (h_{k-j} + h_{j-k})
    # Note that h_{k-j} is non-zero only if 0 <= k-j < N.
    # This is equivalent to the correlation of w and h.
    
    w = np.zeros(2*N - 1)
    w[max_indices] = 1.0
    
    # The gradient of max(conv) wrt h is:
    # grad = correlate(w, h_norm, mode='valid') + correlate(w, h_norm[::-1], mode='valid')
    # Wait, let's verify the derivative.
    # conv_k = sum_{i} h_i h_{k-i}
    # d(conv_k)/d(h_j) = h_{k-j} + h_{j-k} (if j != k-j, i.e., 2j != k)
    # If 2j == k, it is 2 h_j.
    # The formula h_{k-j} + h_{j-k} handles this correctly if we define h_i = 0 for i outside [0, N-1].
    
    # Let's use the property that grad = (w * h) + (w * h)[::-1] restricted to valid parts?
    # Actually, simpler:
    # grad_j = sum_{k} w_k h_{k-j} + sum_{k} w_k h_{j-k}
    
    # Let's compute term1_j = sum_{k} w_k h_{k-j}
    # This is the cross-correlation of w and h?
    # np.correlate(w, h, mode='valid') computes sum_{k} w_k h_{k+j}? No.
    # np.correlate(a, b, mode='valid') at index j is sum_{i} a[i] b[i+j]? 
    # Documentation: c_k = sum_n a_n b_{n+k}.
    # So np.correlate(w, h, 'valid') at index j is sum_{n} w_n h_{n+j}.
    # This is not exactly what we want.
    
    # Let's just loop over max_indices if the count is small, or use FFT if large.
    # Usually max is achieved at a few points.
    # However, for robustness, let's use the full vector operation.
    
    # grad_j = sum_{k=0}^{2N-2} w_k (h_{k-j} + h_{j-k})
    
    # We can compute this by convolving w with h and h[::-1]?
    # Let's stick to a safe loop if count is small, otherwise FFT.
    # Given N=400, a loop over up to 800 indices is fast enough if we vectorize the inner part?
    # No, loop over indices and vectorize the addition.
    
    grad = np.zeros(N)
    
    if len(max_indices) < 50:
        for k in max_indices:
            # Term 1: h_{k-j}
            # Valid j where 0 <= k-j < N => k-N < j <= k
            j_start = max(0, k - N + 1)
            j_end = min(N, k + 1)
            # Indices in h: k-j. Range: k-j_start to k-j_end+1
            # We want to add h[k-j] to grad[j]
            # Let idx_h = k - j
            # j = k - idx_h
            # Range of idx_h: 0 to N-1
            # So we add h[idx_h] to grad[k-idx_h]
            # This is a reversed slice.
            
            # Term 1:
            # grad[j] += h[k-j]
            # Let's create a slice of h and add to grad.
            # j ranges from j_start to j_end-1
            # k-j ranges from k-j_start to k-(j_end-1)
            # h_slice = h[k-j_start : k-j_end+2]
            # grad_slice = grad[j_start : j_end]
            # grad_slice += h_slice[::-1]
            
            h_slice = h_norm[k-j_start : k-j_end+2]
            grad[j_start:j_end] += h_slice[::-1]
            
            # Term 2: h_{j-k}
            # Valid j where 0 <= j-k < N => k < j < k+N
            j_start2 = max(0, k + 1)
            j_end2 = min(N, k + N)
            # j ranges from j_start2 to j_end2-1
            # j-k ranges from j_start2-k to j_end2-1-k
            h_slice2 = h_norm[j_start2-k : j_end2-k]
            grad[j_start2:j_end2] += h_slice2
    else:
        # Fallback to FFT for many max indices (rare)
        # Pad w and h to size 4N
        size = 4*N
        W = np.fft.fft(w, size)
        H = np.fft.fft(h_norm, size)
        # grad_j = sum_k w_k h_{k-j} + sum_k w_k h_{j-k}
        # First term is correlation.
        # Correlation(w, h)_j = sum_k w_k h_{k-j}
        # This is inverse FFT of W * conj(H)? Or W * H?
        # Let's just use np.correlate on padded arrays if needed, but N=400 is small.
        # Actually, let's just use the loop even if 50, it's fast enough in Python if vectorized inside.
        # But 50 * 400 operations is 20k ops, trivial.
        # So the branch above is fine for almost all cases.
        pass
        
    # Project gradient to sum 0 to maintain normalization
    grad_mean = np.mean(grad)
    grad -= grad_mean
    
    # Adaptive step size
    # Simple heuristic: alpha = 1 / (norm(grad) * sqrt(iter))
    # Or fixed small step.
    # Let's use a standard subgradient step with decreasing alpha.
    
    h_new = h_norm - alpha * grad
    
    # Ensure sum is 1 (projection)
    s_new = np.sum(h_new)
    if abs(s_new) > 1e-12:
        h_new /= s_new
    else:
        h_new = h_norm
        
    return h_new

def _diverse_init(max_time=10.0):
    """Generate diverse initial candidates."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(10):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(3):
            if time.time() - t0 > max_time * 0.8:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val

def _subgradient_optimize(h0, max_time=20.0):
    """Optimize using subgradient descent."""
    t0 = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_val = _value(h)
    
    # Initial step size
    alpha = 1.0
    decay = 0.99
    
    while time.time() - t0 < max_time:
        h = _subgradient_step(h, alpha)
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        
        alpha *= decay
        
        # If alpha becomes too small, break
        if alpha < 1e-5:
            break
            
    return best_h, best_val

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Phase 1: Diverse initialization
    h_div, val_div = _diverse_init(max_time=10.0)
    
    best_h = h_div.copy()
    best_val = val_div
    
    # Phase 2: Subgradient optimization
    if time_left() > 20:
        best_h, best_val = _subgradient_optimize(best_h, max_time=min(30.0, time_left() - 10.0))
        
    # Phase 3: Local Refinement with L-BFGS-B on smooth objective (as backup/polish)
    # We can use the smooth objective to polish the subgradient result
    if time_left() > 15:
        try:
            def _smooth_objective(h, beta=100.0):
                s = float(np.sum(h))
                if abs(s) < 1e-12:
                    return 1e18
                conv = np.convolve(h, h)
                m = np.max(conv)
                shifted = beta * (conv - m)
                shifted = np.clip(shifted, -700, 700)
                log_sum = np.log(np.sum(np.exp(shifted)))
                soft_max = m + log_sum / beta
                val = 2.0 * N * soft_max / (s * s)
                return abs(val)

            def _smooth_grad(h, beta=100.0):
                s = float(np.sum(h))
                if abs(s) < 1e-12:
                    return np.zeros(N)
                
                conv = np.convolve(h, h)
                m = np.max(conv)
                shifted = beta * (conv - m)
                shifted = np.clip(shifted, -700, 700)
                weights = np.exp(shifted)
                weights /= np.sum(weights)
                soft_max = m + np.log(np.sum(weights)) / beta
                
                d_soft_max = 2.0 * np.correlate(weights, h, mode='valid')
                
                sign = 1.0
                if (2.0 * N * soft_max / (s * s)) < 0:
                    sign = -1.0
                    
                grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
                return grad

            res = minimize(
                lambda h: _smooth_objective(h, beta=100.0),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=100.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass

    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
