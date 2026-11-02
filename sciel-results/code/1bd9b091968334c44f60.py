import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    # Initial point: uniform
    x = np.ones(N)
    
    def compute_metric(x):
        """Compute the normalized maximum of the autoconvolution."""
        # Autoconvolution via FFT for efficiency
        x_ext = np.zeros(2 * N - 1)
        x_ext[:N] = x
        ac = np.fft.irfft(np.fft.rfft(x_ext) ** 2)[:N]
        # ac[k] is the autocorrelation at lag k (one-sided)
        # Normalize by sum of squares (peak autocorrelation at lag 0)
        # The normalized max could be max(ac)/ac[0] or similar
        # Since for nonneg f, ac[0] = max, this is always 1...
        # Let's try: max of sidelobes relative to peak, or
        # Maybe it's max(ac) / (sum(x))^2 * something
        # Try: max(ac) / ac[0] which is 1 for nonneg
        # Alternative: maybe it's about the discrete autocorrelation normalized by length
        # Let's use: max(ac[1:]) / ac[0] as sidelobe ratio
        # But that's always < 1...
        
        # Actually, re-reading: "normalized maximum of its autoconvolution"
        # Perhaps: max(ac) where ac is computed as convolution (not correlation)
        # For nonnegative, autoconvolution = autocorrelation here
        # Normalized by what? By the integral squared? By the max?
        
        # Let's try the interpretation that gives non-trivial optimization:
        # metric = max_k ac[k] / (sum(x))^2 * N  (normalized by mean energy)
        # For uniform x=1: ac[0] = N, sum(x)=N, so N/N^2*N = 1
        # This is scale-invariant
        
        s = x.sum()
        if s < 1e-10:
            return np.inf
        # Normalize: divide by s^2 / N (mean squared times N)
        # ac[k] / (s^2 / N)
        norm_ac = ac * N / (s * s)
        return float(norm_ac.max())
    
    best_val = compute_metric(x)
    
    def compute_grad(x):
        """Compute gradient of soft-max surrogate for the metric."""
        s = x.sum()
        if s < 1e-10:
            return np.zeros_like(x)
        
        x_ext = np.zeros(2 * N - 1)
        x_ext[:N] = x
        Xf = np.fft.rfft(x_ext)
        ac = np.fft.irfft(Xf * Xf)[:N]
        
        norm_factor = N / (s * s)
        val = ac * norm_factor
        
        # Soft-max with temperature
        temp = 1.0
        scaled = val / temp
        max_scaled = scaled.max()
        exps = np.exp(scaled - max_scaled)
        weights = exps / exps.sum()  # softmax weights
        
        # Gradient of max(val) approx by weighted sum
        # d/max(val) / dx_j = sum_k weights_k * d(ac[k]) / dx_j * norm_factor
        # d(ac[k]) / dx_j for autoconvolution
        # ac[k] = sum_i x[i] x[i+k] (for k >= 0)
        # d(ac[k])/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if k > 0 and j-k >= 0)
        # More carefully: ac[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
        # d(ac[k])/dx[j] = x[j+k] (if j+k <= N-1) + x[j-k] (if k>0 and j-k >= 0)
        
        # Use FFT-based gradient: ac = irfft(|F|^2), so grad is convolve-like
        # Simpler: direct computation of weighted gradient
        grad = np.zeros(N)
        for k in range(N):
            w = weights[k] * norm_factor
            if w < 1e-15:
                continue
            if k == 0:
                # d(ac[0])/dx[j] = 2*x[j]
                grad += 2 * w * x
            else:
                # d(ac[k])/dx[j] = x[j+k] (j+k<N) + x[j-k] (j-k>=0)
                # x[j+k] contributes to positions j where j+k < N, i.e., j < N-k
                # This is x shifted by -k (for the first term) 
                # and x shifted by +k (for the second term)
                # d(ac[k])/dx[j] = x[j+k] + x[j-k] (with boundary handling)
                if N - k > 0:
                    # x[j+k] for j in [0, N-k-1] -> this is x[k:] placed at [0, N-k-1]
                    grad[:N-k] += w * x[k:]
                if k < N:
                    # x[j-k] for j in [k, N-1] -> this is x[:N-k] placed at [k, N-1]
                    grad[k:] += w * x[:N-k]
        
        # Also gradient of the normalization factor N/(s^2)
        # d(N/s^2)/dx[j] = -2N/s^3
        dnorm = -2 * N / (s**3)
        # val = ac * N/s^2, so dval/dx[j] includes ac * d(N/s^2)/dx[j] + (N/s^2) * dac/dx[j]
        # We already computed (N/s^2)*dac/dx[j] as 'grad'
        # Add ac * dnorm contribution, weighted by softmax
        ac_scaled = val  # already = ac * N/s^2
        # The full gradient: sum_k w_k * (dac_k/dx * N/s^2 + ac_k * dN/s^2/dx)
        # dN/s^2/dx[j] = -2N/s^3, so ac_k * (-2N/s^3) * ... but this is same for all k
        # So: grad += (-2N/s^3) * sum_k w_k * ac_k * ones
        sum_w_ac = np.sum(weights * ac)
        grad += dnorm * sum_w_ac * np.ones(N)
        
        return grad
    
    # Adam optimizer
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    patience = 0
    best_x = x.copy()
    best_val_now = best_val
    
    while time.time() - start_time < DEADLINE:
        t += 1
        g = compute_grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project: keep non-negative and same sum (scale invariance)
        x_new = np.maximum(x_new, 0.0)
        if x_new.sum() > 1e-10:
            x_new = x_new / x_new.sum() * x.sum()  # preserve sum
        else:
            break
        
        val_new = compute_metric(x_new)
        
        if val_new < best_val_now - 1e-10:
            best_val_now = val_new
            best_x = x_new.copy()
            patience = 0
        else:
            patience += 1
        
        x = x_new
        
        # Reduce learning rate if stuck
        if patience % 100 == 0 and patience > 50:
            lr *= 0.9
            patience = 0
        
        # Safety check
        if not np.isfinite(val_new):
            x = best_x.copy()
            break
    
    # Final polish: try a few different smooth surrogates
    result = best_x
    s = result.sum()
    if s < 1e-10:
        return [1.0] * N
    result = result / s  # normalize
    
    # Ensure non-negative and return as list
    result = np.maximum(result, 0.0)
    return list(result)

