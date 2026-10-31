import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Log-sum-exp smooth approximation of max
    def smooth_max(c, temp=0.01):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
    
    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        c = np.convolve(h, h, mode='full')
        return smooth_max(c, temp) / (s * s)
    
    # Evaluate uniform baseline
    h_uniform = np.ones(N)
    best_val = metric(h_uniform)
    best_h = h_uniform.copy()
    
    # Phase 1: Try structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper (reduces edge effects in convolution)
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Sinc-like (flat top with ringing)
    n = np.arange(N)
    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)
    
    # Gaussian
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Raised cosine (Blackman window)
    n = np.arange(N)
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning window
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top window
    flat_top = 0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) + 0.277263158 * np.cos(4*np.pi*n/(N-1)) + 0.083578947 * np.cos(6*np.pi*n/(N-1)) + 0.006947368 * np.cos(8*np.pi*n/(N-1))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau: high at edges, low in center (to reduce center peak)
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse: low at edges, high in center
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Evaluate candidates
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization with smooth max
    # Use log-sum-exp with adaptive temperature for better convergence
    
    def objective(x, temp=0.005):
        if elapsed() > DEADLINE - 10:
            return 1e10
        # Convert to positive heights
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)
    
    # Start from best candidate
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)  # normalize
    
    # Try multiple temperatures for smoother optimization
    for temp in [0.01, 0.005, 0.002]:
        if elapsed() > DEADLINE - 15:
            break
        try:
            result = minimize(
                lambda x: objective(x, temp), x0, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 400}
            )
            if result.success or result.fun < best_val:
                h_opt = np.exp(result.x - np.max(result.x))
                v = metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)  # update starting point
        except Exception:
            pass
    
    # Phase 3: Coordinate descent with exact max (not smooth)
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    # Use FFT-based convolution for faster computation
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        # Use FFT for autoconvolution
        nfft = 2 * N
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        # The linear convolution is in the first 2N-1 elements
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    # Simpler: just recompute conv for each coordinate update
    no_improve = 0
    max_no_improve = 2000
    
    while elapsed() < DEADLINE - 20 and no_improve < max_no_improve:
        # Random coordinate
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        # Try several candidate values with finer granularity
        mean_h = np.mean(current_h)
        # Focus on local changes around current value
        step_factors = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
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
        
        # Periodic perturbation with smaller steps
        if no_improve % 150 == 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
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
    
    # Phase 4: Fine-tune with small steps using FFT
    if elapsed() < DEADLINE - 5:
        step_frac = 0.005
        for _ in range(800):
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
