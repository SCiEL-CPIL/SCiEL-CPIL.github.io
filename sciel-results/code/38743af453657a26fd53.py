import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    dx = 1.0 / (2.0 * N)
    
    # Initial guess: Raised Cosine is a good start, close to uniform but with some variation
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx / 2
    f = 1.0 + 0.5 * np.cos(2 * np.pi * x)
    f = np.maximum(f, 0.0)
    integral = np.sum(f) * dx
    f = f / integral
    
    best_f = f.copy()
    
    def compute_autoconv(f):
        """Compute linear autoconvolution using FFT."""
        fft_len = 2 * N
        fft_f = np.fft.fft(f, n=fft_len)
        autoconv_freq = np.abs(fft_f) ** 2
        autoconv_full = np.real(np.fft.ifft(autoconv_freq)) * dx
        return autoconv_full

    def compute_score(f):
        """Compute the normalized peak of autoconvolution."""
        autoconv = compute_autoconv(f)
        integral = np.sum(f) * dx
        norm = integral ** 2
        if norm < 1e-15:
            return np.inf, np.inf, 0
        peak_val = np.max(autoconv)
        peak_idx = np.argmax(autoconv)
        normalized_peak = peak_val / norm
        return normalized_peak, peak_val, peak_idx

    best_score, _, _ = compute_score(f)
    
    # Adam optimizer parameters
    lr = 0.005
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    
    # Regularization weight for L2 norm of autoconv
    reg_weight = 0.1
    
    for step in range(50000):
        if time.time() > deadline - 3.0:
            break
        
        t += 1
        integral = np.sum(f) * dx
        
        autoconv = compute_autoconv(f)
        peak_val = np.max(autoconv)
        peak_idx = np.argmax(autoconv)
        
        norm = integral ** 2
        current_score = peak_val / norm if norm > 1e-15 else np.inf
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
        
        # Compute Gradient
        # 1. Subgradient of the peak
        # d(max_k C(k))/df[i] = dC(k*)/df[i] = 2 * f[k* - i] (with appropriate shifting)
        # In linear convolution C(k) = sum_j f[j] f[k-j]
        # dC(k)/df[i] = f[k-i] (from j=i) + f[k-i] (from j=k-i)
        # Note: f is defined on indices 0..N-1.
        # We need to map peak_idx to the linear convolution index space.
        # The FFT output `autoconv_full` has length 2N.
        # The linear convolution of two length N sequences has length 2N-1.
        # The FFT result wraps.
        # For zero-padded FFT of length 2N:
        # autoconv_full[k] corresponds to the circular convolution.
        # Since we padded to 2N, and the signal is length N,
        # indices 0 to N-1 in autoconv_full correspond to lags 0 to N-1?
        # Actually, np.fft.ifft of |fft(f, 2N)|^2 gives the linear convolution
        # placed in indices 0 to 2N-2, with the rest being 0 or wrap-around artifacts.
        # Specifically, for real f, the convolution is symmetric around N-1?
        # Let's verify the shift.
        # If f is a delta at 0, f*f is delta at 0.
        # fft(f) is 1. |fft(f)|^2 is 1. ifft is delta at 0.
        # So peak_idx should be 0 for a delta at 0.
        # If f is delta at N/2, f*f is delta at N.
        # So peak_idx will be N.
        
        # The gradient w.r.t f[i] of C(peak_idx) is:
        # g[i] = 2 * f[peak_idx - i] if 0 <= peak_idx - i < N
        #       + f[peak_idx - i + 2N] ... (wrap around for circular, but we want linear)
        # Since we are doing linear convolution effectively (via zero padding),
        # we should only take the valid terms.
        
        g_sub = np.zeros(N)
        # Vectorized shift
        # We need f[peak_idx - i]
        # Let k = peak_idx
        # g[i] = 2 * f[k - i]
        # This is a shifted version of f.
        # If k - i is in [0, N-1], it's valid.
        
        # Create an index array
        idx = np.arange(N)
        target_idx = peak_idx - idx
        # Wrap around for the FFT representation?
        # No, for linear convolution, if index is out of bounds, it's 0.
        # But our FFT calculation `autoconv_full` *is* the linear convolution
        # embedded in a 2N array.
        # The relationship between `autoconv_full` indices and linear conv indices:
        # Linear conv index l (0 to 2N-2) maps to autoconv_full[l] for l < N?
        # And autoconv_full[2N - l] for l > N?
        # Actually, np.fft.ifft of |fft(f, 2N)|^2 gives the circular convolution.
        # Because f is zero-padded to 2N, the circular convolution of length 2N
        # is identical to the linear convolution of length 2N-1, followed by 0,
        # PROVIDED that the wrap-around part doesn't overlap.
        # The linear conv has support [0, 2N-2].
        # The circular conv has support [0, 2N-1].
        # Indices 0 to 2N-2 are correct. Index 2N-1 is 0 (or sum of wrap).
        # So autoconv_full[0:2N-1] is the linear convolution.
        
        # So peak_idx is the index in the linear convolution.
        # d C(peak_idx) / d f[i] = f[peak_idx - i] + f[peak_idx - i] ?
        # C(k) = sum_j f[j] f[k-j]
        # d/d f[i]:
        # Term 1: j=i -> f[i] f[k-i] -> derivative is f[k-i]
        # Term 2: k-j=i -> j=k-i -> f[k-i] f[i] -> derivative is f[i]
        # So g[i] = f[k-i] + f[i] ? No.
        # C(k) = sum_{j} f_j f_{k-j}
        # d C(k) / d f_i = f_{k-i} (from j=i) + f_{k-i} (from j=k-i, term is f_{k-i} f_i)
        # Wait, if j=k-i, the term is f_{k-i} f_i. Derivative w.r.t f_i is f_{k-i}.
        # So yes, g[i] = 2 f[k-i] if i != k-i.
        # If i == k-i (2i=k), the term is f_i^2, derivative is 2 f_i.
        # So g[i] = 2 f[k-i] is generally correct, assuming f is defined such that
        # f[k-i] is 0 if out of bounds.
        
        # We need to extract f[k-i] for all i.
        # Let's use np.roll or indexing.
        # f_shifted[i] = f[k - i]
        # This is f reversed and shifted.
        
        # Efficient way:
        # f_rev = f[::-1]
        # f[k-i] is element (k-i) of f.
        # We can create an array of indices: k - np.arange(N)
        # Mask valid indices.
        
        valid_mask = (target_idx >= 0) & (target_idx < N)
        g_sub[valid_mask] = 2.0 * f[target_idx[valid_mask]]
        
        # 2. Regularization: Gradient of L2 norm of autoconvolution
        # Minimize sum(autoconv^2)
        # This is equivalent to minimizing sum |FFT(f)|^4
        F = np.fft.fft(f, n=2*N)
        absF_sq = np.abs(F) ** 2
        G = absF_sq * F
        grad_l2 = 4.0 * np.real(np.fft.ifft(G, n=2*N)[:N])
        grad_l2 = grad_l2 * (2.0 * N / N) # Scaling factor for zero-padded FFT
        
        # Combine gradients
        # We want to minimize peak_val / integral^2
        # Approx gradient: g_sub / integral^2
        # Regularization helps flatten the spectrum, preventing spikes elsewhere
        grad = g_sub / (integral ** 2) + reg_weight * grad_l2 / (integral ** 2)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        f_new = f - update
        f_new = np.maximum(f_new, 0.0)
        
        # Renormalize to keep integral constant (scale invariance)
        new_int = np.sum(f_new) * dx
        if new_int > 1e-12:
            f_new = f_new * (integral / new_int)
        
        f = f_new
    
    return list(best_f)
