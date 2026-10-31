import numpy as np
import time

N = 600
DEADLINE = 135  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def compute_conv_max(h):
        """Compute max of linear autoconvolution of h."""
        c = np.convolve(h, h, mode='full')
        return np.max(c)
    
    def metric(h):
        """Normalized metric: max_conv / (sum_h)^2, scale-invariant."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        return compute_conv_max(h) / (s * s)
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 2 * N
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def fast_conv(h):
        """Compute full autoconvolution using FFT."""
        nfft = 2 * N
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1]
    
    # Chebyshev/cosine parameterization: h[i] = c0 + sum_{k=1}^{K} c_k * cos(2*pi*k*i/N)
    K = 12  # number of cosine harmonics
    
    def cheb_to_h(coeffs):
        """Convert cosine series coefficients to step heights (nonnegative)."""
        i = np.arange(N)
        h = np.full(N, coeffs[0])
        for k in range(1, K+1):
            h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
        return np.maximum(h, 0.0)
    
    # Phase 1: Evaluate structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    i = np.arange(N)
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
    
    # Two-plateau
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse plateau
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Triangle (peak in center)
    tri = np.ones(N)
    center = N // 2
    for idx in range(N):
        dist = abs(idx - center)
        tri[idx] = max(0.1, 1.0 - dist / (N/2))
    candidates.append(tri)
    
    # Super-Gaussian
    for sigma in [0.15, 0.2, 0.25, 0.35]:
        xg = (i - N/2) / (N * sigma)
        candidates.append(np.exp(-0.5 * xg**4))
    
    best_val = 1e10
    best_h = np.ones(N)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Remez-like iterative equioscillation on cosine coefficients
    # (vectorized gradient computation)
    
    def fit_cosine(h, K_order):
        """Fit first K_order cosine coefficients to h."""
        i = np.arange(N)
        A = np.zeros((N, K_order + 1))
        A[:, 0] = 1.0
        for k in range(1, K_order + 1):
            A[:, k] = np.cos(2 * np.pi * k * i / N)
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs
    
    coeffs = fit_cosine(best_h, K)
    
    # Precompute cosine basis matrix for fast gradient
    i_arr = np.arange(N)
    cos_basis = np.zeros((K + 1, N))
    cos_basis[0] = 1.0
    for k in range(1, K + 1):
        cos_basis[k] = np.cos(2 * np.pi * k * i_arr / N)
    
    def get_conv_from_coeffs(coeffs):
        h = cheb_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(2*N-1), 0
        return fast_conv(h), s
    
    for remez_iter in range(50):
        if elapsed() > DEADLINE - 30:
            break
        
        h = cheb_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-15:
            break
        
        c = fast_conv(h)
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        
        # Vectorized gradient computation
        # c[k] = sum_i h[i]*h[i+k] for k < N (with appropriate bounds)
        # dc[k]/dcoeffs[j] = sum_i (cos_basis[j,i]*h[i+k] + h[i]*cos_basis[j,i+k])
        k = max_idx
        if k >= N:
            k = 2*N - 1 - k
        
        if k == 0:
            # c[0] = sum h[i]^2, dc[0]/dcoeffs[j] = 2 * sum h[i] * cos_basis[j,i]
            grad = 2.0 * cos_basis @ h / (s * s)
        else:
            # c[k] = sum_{i=0}^{N-k-1} h[i]*h[i+k] + sum_{i=k}^{N-1} h[i]*h[i-k]
            # = sum_{i=0}^{N-1} h[i]*h[(i+k)%N] ... no, it's linear conv
            # For linear conv: c[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k+N-1)} ... 
            # Actually c[k] = sum_{i} h[i] * h[i+k] where both indices in [0,N-1]
            # So for 0 <= k < N: c[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
            # For N <= k < 2N-1: c[k] = sum_{i=k-N+1}^{N-1} h[i]*h[i-(k-N+1)]
            # By symmetry c[k] = c[2N-1-k], so we only need k < N
            
            n_valid = N - k
            h1 = h[:n_valid]
            h2 = h[k:k+n_valid]
            c1 = cos_basis[:, :n_valid]  # shape (K+1, n_valid)
            c2 = cos_basis[:, k:k+n_valid]
            # grad[j] = sum_i (c1[j,i]*h2[i] + h1[i]*c2[j,i])
            grad = (c1 @ h2 + c2 @ h1) / (s * s)
        
        # Simple gradient step to reduce the peak
        grad_norm = np.max(np.abs(grad)) + 1e-15
        lr = 0.01 * max_val / grad_norm
        lr = min(lr, 0.1)
        
        new_coeffs = coeffs - lr * grad
        new_h = cheb_to_h(new_coeffs)
        
        if np.sum(new_h) > 1e-15:
            new_val = fast_metric(new_h)
            if new_val < best_val:
                best_val = new_val
                best_h = new_h.copy()
                coeffs = new_coeffs
            else:
                new_coeffs = coeffs - lr * 0.1 * grad
                new_h = cheb_to_h(new_coeffs)
                if np.sum(new_h) > 1e-15:
                    new_val = fast_metric(new_h)
                    if new_val < best_val:
                        best_val = new_val
                        best_h = new_h.copy()
                        coeffs = new_coeffs
    
    # Phase 2b: Water-filling iteration
    # Idea: where the local autoconvolution contribution is highest, 
    # the height should be lower (to flatten the convolution).
    # Where it's low, we can raise the height.
    # This is analogous to water-filling in information theory.
    
    current_h = best_h.copy()
    current_val = best_val
    
    for wf_iter in range(200):
        if elapsed() > DEADLINE - 25:
            break
        
        c = fast_conv(current_h)
        s = np.sum(current_h)
        if s < 1e-15:
            break
        
        # The "interference" at position i is the contribution to convolution peaks
        # from h[i]. We want to identify where increasing h[i] most increases max conv.
        # Approximate: the local "peak pressure" at position i is the max of
        # c[k] where i contributes, weighted by h[i].
        
        # Simpler approach: compute the "sensitivity" of max_conv to each h[i]
        # Sensitivity_i = max over k of (h[i+k] if i+k < N else 0) + (h[i-k] if i-k >= 0 else 0)
        # This is the derivative of c[k] w.r.t. h[i] at the peak lag
        
        max_idx = np.argmax(c)
        k_peak = max_idx
        if k_peak >= N:
            k_peak = 2*N - 1 - k_peak
        
        # Sensitivity: how much does c[k_peak] change if we change h[i]?
        # c[k_peak] = sum_{j=0}^{N-1-k_peak} h[j]*h[j+k_peak]
        # dc/dh[i] = h[i+k_peak] (if i+k_peak < N) + h[i-k_peak] (if i-k_peak >= 0)
        sensitivity = np.zeros(N)
        n_valid = N - k_peak
        sensitivity[:n_valid] += current_h[k_peak:k_peak+n_valid]
        if k_peak > 0:
            sensitivity[k_peak:] += current_h[:n_valid]
        
        # Water-filling: raise heights where sensitivity is low, lower where high
        # Use a soft version: h_new[i] = h[i] + alpha * (target - sensitivity[i])
        # where target is the mean sensitivity
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-10
        
        # Normalize sensitivity to [0, 1]
        sens_norm = (sensitivity - np.min(sensitivity)) / (np.max(sensitivity) - np.min(sensitivity) + 1e-10)
        
        # Water-filling step: increase where sens_norm is low, decrease where high
        alpha = 0.05
        adjustment = alpha * (0.5 - sens_norm) * np.mean(current_h)
        new_h = current_h + adjustment
        new_h = np.maximum(new_h, 0.0)
        
        if np.sum(new_h) < 1e-15:
            break
        
        new_val = fast_metric(new_h)
        if new_val < current_val:
            current_val = new_val
            current_h = new_h
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            # Try smaller step
            new_h2 = current_h + 0.1 * adjustment
            new_h2 = np.maximum(new_h2, 0.0)
            if np.sum(new_h2) > 1e-15:
                new_val2 = fast_metric(new_h2)
                if new_val2 < current_val:
                    current_val = new_val2
                    current_h = new_h2
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 3: Direct coordinate descent on the full 600-dim space
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(42)
    
    no_improve = 0
    max_no_improve = 5000
    
    while elapsed() < DEADLINE - 15 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        candidates_val = list(orig + step_sizes) + list(orig - step_sizes) + [0.0, orig]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-10:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1
        
        # Periodic perturbation
        if no_improve % 200 == 0 and no_improve > 0:
            pert = current_h.copy()
            pert += 0.03 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 4: Fine-tune with paired updates
    if elapsed() < DEADLINE - 5:
        step_frac = 0.003
        for _ in range(2000):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    # Final verification with exact metric
    final_val = metric(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h.tolist()
