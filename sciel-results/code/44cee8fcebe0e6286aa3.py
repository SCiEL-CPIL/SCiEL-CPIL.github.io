import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145  # Safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max autoconvolution using FFT."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def metric(h):
        """Exact normalized max autoconvolution."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        c = np.convolve(h, h, mode='full')
        return np.max(c) / (s * s)
        
    def lse_metric(h, temp=0.05):
        """Log-sum-exp approximation of the max autoconvolution."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        
        c_max = np.max(c)
        # Numerical stability
        c_shifted = (c - c_max) / temp
        exp_c = np.exp(c_shifted)
        lse_val = temp * (np.log(np.sum(exp_c)) + c_max / temp)
        return lse_val / (s * s)
        
    def lse_metric_grad(h, temp=0.05):
        """Gradient of the LSE metric w.r.t. h."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
            
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        
        c_max = np.max(c)
        c_shifted = (c - c_max) / temp
        exp_c = np.exp(c_shifted)
        sum_exp = np.sum(exp_c)
        p = exp_c / sum_exp  # Softmax weights
        
        lse_val = temp * (np.log(sum_exp) + c_max / temp)
        
        # Gradient of LSE w.r.t. c is p
        # Gradient of c w.r.t. h
        # c_k = sum_j h_j h_{j+k}
        # dc_k/dh_i = h_{i+k} + h_{i-k} (with bounds)
        # We need sum_k p_k (h_{i+k} + h_{i-k})
        
        # Let's compute this via FFT
        # P = fft(p)
        # Term 1: sum_k p_k h_{i+k} -> This is correlation of p and h, or convolution with reversed p
        # Term 2: sum_k p_k h_{i-k} -> This is convolution of p and h
        
        # Since p is defined on 0..2N-2, we need to be careful with indices.
        # However, h is defined on 0..N-1.
        # The convolution c is linear.
        
        # Let's use the property that for linear convolution, the gradient can be computed as:
        # g = ifft(fft(p_padded) * fft(h_padded)) + ifft(fft(p_rev_padded) * fft(h_padded))
        
        # p is of length 2N-1. h is of length N.
        # We pad both to nfft.
        
        p_pad = np.zeros(nfft)
        p_pad[:len(p)] = p
        
        h_pad = np.zeros(nfft)
        h_pad[:N] = h
        
        P = np.fft.rfft(p_pad)
        H_fft = np.fft.rfft(h_pad)
        
        # Convolution p * h
        conv1 = np.fft.irfft(P * H_fft, n=nfft)[:N]
        
        # Correlation p * h (which is conv of reversed p and h)
        # Reversing p: p_rev[k] = p[len(p)-1-k]
        p_rev = p[::-1]
        p_rev_pad = np.zeros(nfft)
        # Shift p_rev so that it aligns correctly for the "i+k" term
        # Actually, sum_k p_k h_{i+k} is (p_{-} * h)_i where p_{-}[k] = p_{-k}.
        # In discrete terms with 0-based indexing for p (0 to 2N-2), p_{-k} corresponds to index (2N-2-k).
        # So p_rev is correct.
        p_rev_pad[:len(p_rev)] = p_rev
        
        P_rev = np.fft.rfft(p_rev_pad)
        conv2 = np.fft.irfft(P_rev * H_fft, n=nfft)[:N]
        
        grad_lse_h = conv1 + conv2
        
        # Gradient of metric = grad_lse_h / s^2 - 2 * lse_val * h / s^3
        grad_metric = grad_lse_h / (s * s) - 2 * lse_val * h / (s * s * s)
        
        return lse_val / (s * s), grad_metric

    # Phase 1: Generate and evaluate structured candidates
    i = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    candidates.append(0.5 * (1 + np.cos(np.pi * i / (N - 1))))
    
    # Gaussian
    xg = (i - N/2) / (N * 0.3)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Sparse random patterns
    np.random.seed(42)
    for density in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sparse = (np.random.rand(N) < density).astype(float)
        candidates.append(sparse)
        
    # Low-frequency smooth patterns
    for freq in [1, 2, 3, 4, 5]:
        smooth_pat = 0.5 + 0.5 * np.cos(2 * np.pi * freq * i / N)
        candidates.append(smooth_pat)
        
    best_val = 1e10
    best_h = np.ones(N)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
            
    # Phase 2: Gradient-based optimization using LSE surrogate
    # We optimize in log-space to ensure non-negativity
    def objective_log(x):
        if elapsed() > DEADLINE - 20:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return lse_metric(h, temp=0.05)
        
    def objective_log_grad(x):
        if elapsed() > DEADLINE - 20:
            return 1e10, np.zeros(N)
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        val, grad_h = lse_metric_grad(h, temp=0.05)
        
        # Chain rule: d/dx_i = d/dh_i * h_i
        grad_x = grad_h * h
        # Center the gradient to handle the shift invariance
        grad_x = grad_x - np.mean(grad_x)
        
        return val, grad_x
        
    # Try optimizing from the best candidate and a few others
    start_points = [best_h]
    # Add a few random starts
    for _ in range(3):
        rand_h = np.random.rand(N)
        rand_h = np.clip(rand_h, 1e-6, None)
        start_points.append(rand_h)
        
    for h_init in start_points:
        if elapsed() > DEADLINE - 30:
            break
            
        x0 = np.log(h_init + 1e-15)
        x0 = x0 - np.max(x0)
        
        try:
            result = minimize(
                objective_log, x0, jac=objective_log_grad, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            if result.fun < best_val:
                h_opt = np.exp(result.x - np.max(result.x))
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Polishing with exact metric
    # Use subgradient descent on the exact metric
    h_curr = best_h.copy()
    val_curr = fast_metric(h_curr)
    
    for polish_iter in range(50):
        if elapsed() > DEADLINE - 10:
            break
            
        s = np.sum(h_curr)
        c = np.convolve(h_curr, h_curr, mode='full')
        max_val = np.max(c)
        max_indices = np.where(c == max_val)[0]
        
        # Subgradient of max(c)
        # Average of gradients at max indices
        grad_max = np.zeros(N)
        for k in max_indices:
            # dc_k/dh_i = h_{i+k} + h_{i-k}
            for i in range(N):
                if i + k < N:
                    grad_max[i] += h_curr[i+k]
                if i - k >= 0:
                    grad_max[i] += h_curr[i-k]
        grad_max /= len(max_indices)
        
        # Gradient of V = max(c)/s^2
        grad_V = grad_max / (s * s) - 2 * max_val * h_curr / (s * s * s)
        
        # Adaptive step size
        # Try a few step sizes
        best_step = 0
        best_val_step = val_curr
        for lr in [0.01, 0.005, 0.001, 0.0005]:
            new_h = h_curr - lr * grad_V
            new_h = np.clip(new_h, 0, None)
            if np.sum(new_h) < 1e-15:
                continue
            new_val = fast_metric(new_h)
            if new_val < best_val_step:
                best_val_step = new_val
                best_step = lr
                best_h_step = new_h
                
        if best_step > 0:
            h_curr = best_h_step
            val_curr = best_val_step
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
        else:
            break
            
    # Final check
    final_val = metric(best_h)
    if final_val < best_val:
        best_val = final_val
        
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
        
    return best_h.tolist()
