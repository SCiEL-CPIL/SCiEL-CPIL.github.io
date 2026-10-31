import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024  # next power of 2 >= 2*N-1 = 1199
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def get_autoconv(h):
        """Return the raw autoconvolution array."""
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1]
    
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
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)
    
    # Evaluate uniform baseline
    h_uniform = np.ones(N)
    best_val = fast_metric(h_uniform)
    best_h = h_uniform.copy()
    
    # Phase 1: Try structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Sinc-like
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
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning window
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top window
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*n/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*n/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau: high at edges, low in center
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse: low at edges, high in center
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Triangular (Bartlett)
    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)
    
    # Evaluate candidates
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization with smooth max, annealed temperatures
    def objective(x, temp=0.01):
        if elapsed() > DEADLINE - 15:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)
    
    def objective_grad(x, temp=0.01):
        """Objective with analytic gradient for L-BFGS-B."""
        if elapsed() > DEADLINE - 15:
            return 1e10, np.zeros_like(x)
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros_like(x)
        
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        
        # Smooth max approximation
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        weights = np.exp(c_shifted) / np.sum(np.exp(c_shifted))
        
        # Gradient of smooth_max(c) w.r.t. c
        grad_c = weights
        
        # Gradient of c w.r.t. h: c[k] = sum_i h[i]*h[i+k]
        # dc/dh[j] = sum_k grad_c[k] * (h[j-k] + h[j+k]) (with boundary handling)
        # Using FFT: autocorr = ifft(|fft(h)|^2)
        # d(autocorr)/dh = 2 * autocorr convolved with delta... 
        # Actually for autoconvolution c[k] = sum_i h[i]h[i+k], 
        # the gradient w.r.t h[j] is: sum_k grad_c[k] * (h[j-k] + h[j+k]) where valid
        # This is equivalent to: 2 * (grad_c * h) autocorrelation-like
        
        # More carefully: c = autocorr(h), so dc/dh[j] = sum_k (grad_c[k] * h[j-k] + grad_c[k] * h[j+k])
        # = 2 * sum_k grad_c[k] * h[j-k] (for symmetric, but autoconv is not quite symmetric in gradient)
        # Let's use: d/dh_j [sum_k grad_c[k] * sum_i h[i]h[i+k]] 
        # = sum_k grad_c[k] * (sum_i [delta_{i,j} h[i+k] + h[i] delta_{i+k,j}])
        # = sum_k grad_c[k] * (h[j+k] + h[j-k])
        
        # Compute via FFT: let g = grad_c (padded), then result = ifft(fft(h) * fft(g)) + ifft(fft(h) * fft(g_flipped))
        # Simpler: use direct for 600 elements? No, use FFT.
        
        # c[k] = sum_i h[i] h[i+k] for k=0..2N-2
        # This is circular autocorrelation if we pad. Let's use linear.
        # For gradient: dh_j = sum_k grad_c[k] (h[j-k] + h[j+k]) with bounds
        
        # Use FFT trick: 
        # autocorr(h)[k] = sum_i h[i]h[i+k]
        # d(autocorr(h)[k])/dh[j] = h[j+k] + h[j-k] (with h[i]=0 outside [0,N-1])
        # So grad_h[j] = sum_k grad_c[k] (h[j+k] + h[j-k])
        # = (grad_c * h)[j] + (grad_c * h_rev)[j] where * is correlation
        
        # Let's compute: A[j] = sum_k grad_c[k] * h[j+k] = correlation of grad_c and h
        # B[j] = sum_k grad_c[k] * h[j-k] = cross-correlation
        
        # Using FFT: corr(a,b)[j] = sum_k a[k] b[j+k] = ifft(fft(a) * conj(fft(b)))... 
        # Actually it's easier to just do it with np.convolve for 600 elements
        
        # grad_h[j] = sum_k grad_c[k] h[j+k] + sum_k grad_c[k] h[j-k]
        # First term: (grad_c ⊛ h)[j] where ⊛ is cross-correlation
        # = convolve(grad_c, h)[j + len(grad_c) - 1] roughly
        
        # Let me just compute directly since N=600, 2N-1=1199, this is O(N^2) = 360K which is fine
        # But we're in a gradient loop, so let's be smarter.
        
        # Actually for 600 elements, direct computation of gradient might be fast enough
        # grad_h = np.zeros(N)
        # for k in range(2*N-1):
        #     if grad_c[k] != 0:
        #         # h[j+k] term: j from 0 to N-1-k
        #         if k < N:
        #             grad_h[:N-k] += grad_c[k] * h[k:]
        #             # h[j-k] term: j from k to N-1
        #             grad_h[k:] += grad_c[k] * h[:N-k]
        
        # This is O(N^2) per gradient eval. With 600 elements, 360K ops. Acceptable.
        
        grad_h = np.zeros(N)
        gc = grad_c
        for k in range(min(2*N-1, N)):
            if abs(gc[k]) > 1e-12:
                if k < N:
                    grad_h[:N-k] += gc[k] * h[k:]
                    grad_h[k:] += gc[k] * h[:N-k]
        
        # Now account for the chain rule: h = exp(x - max(x))
        # dh/dx[j] = h[j] * (delta[j] - 1/N) approximately (since max is piecewise)
        # For simplicity, use dh/dx[j] ≈ h[j] (ignoring the max subtraction, which is a constant shift)
        # Actually x_shifted = x - max(x), so h = exp(x_shifted)
        # dh[j]/dx[i] = h[j] * (delta[j,i] - 1/N) if we treat max as constant
        # The 1/N term contributes a constant to all grad, so for optimization it's fine to use just h[j]
        
        # Objective = smooth_max(c) / s^2
        # d(obj)/dx[j] = (grad_c_smooth dot dc/dh) / s^2 * dh/dx[j] - 2 * smooth_max(c) / s^3 * (sum dh/dx[j])
        # sum dh/dx[j] = sum h[j] = s (approximately, ignoring max shift)
        
        smooth_max_val = temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
        obj = smooth_max_val / (s * s)
        
        # Gradient w.r.t h:
        grad_h_obj = grad_h / (s * s) - 2 * smooth_max_val * grad_h * s / (s * s * s)
        # Wait: d/dh [smooth_max(c)/s^2] = (d(smooth_max(c))/dh * s^2 - smooth_max(c) * 2*s * ds/dh) / s^4
        # ds/dh[j] = 1
        # = grad_h / s^2 - 2 * smooth_max(c) / (s^3)
        
        grad_h_obj = grad_h / (s * s) - 2 * smooth_max_val / (s ** 3)
        
        # Chain rule to x: dx/dh... h = exp(x - max(x))
        # For simplicity, multiply by h (Jacobian of exp is exp itself, ignoring max)
        grad_x = grad_h_obj * h
        
        return obj, grad_x
    
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)
    
    # Anneal from high to low temperature for better convergence
    temps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for temp in temps:
        if elapsed() > DEADLINE - 20:
            break
        try:
            result = minimize(
                lambda x: objective_grad(x, temp), x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
            )
            if result.fun < 1e9:
                h_opt = np.exp(result.x - np.max(result.x))
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)
        except Exception:
            pass
    
    # Phase 3: Coordinate descent with FFT-based metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    no_improve = 0
    max_no_improve = 5000
    
    while elapsed() < DEADLINE - 30 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
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
        
        # Periodic perturbation to escape local minima
        if no_improve % 200 == 0:
            pert = current_h.copy()
            pert += 0.08 * np.random.randn(N) * np.mean(current_h)
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
    
    # Phase 4: Peak-targeted polish - identify where autoconvolution peaks and reduce them
    if elapsed() < DEADLINE - 10:
        h_curr = best_h.copy()
        val_curr = best_val
        c = get_autoconv(h_curr)
        s = np.sum(h_curr)
        
        # Find top peak locations in autoconvolution
        top_k = 10
        peak_lags = np.argsort(c)[::-1][:top_k]
        
        for _ in range(500):
            if elapsed() > DEADLINE - 8:
                break
            
            # Pick a random peak lag
            lag = peak_lags[np.random.randint(len(peak_lags))]
            
            # For lag L, autoconv[L] = sum_i h[i]*h[i+L]
            # To reduce this, we can try decreasing h[i] or h[i+L] for large contributors
            # Try small adjustments to indices that contribute to this peak
            valid_pairs = []
            for i in range(N - lag):
                if h_curr[i] > 1e-6 and h_curr[i+lag] > 1e-6:
                    valid_pairs.append((i, i+lag))
            
            if not valid_pairs:
                continue
            
            # Pick a few random contributing pairs
            n_try = min(5, len(valid_pairs))
            trial_pairs = np.random.choice(len(valid_pairs), n_try, replace=False)
            
            improved = False
            for ti in trial_pairs:
                i, j = valid_pairs[ti]
                # Try reducing one of the pair
                for (idx, factor) in [(i, 0.95), (j, 0.95)]:
                    new_h = h_curr.copy()
                    new_h[idx] *= factor
                    if new_h[idx] < 0:
                        continue
                    v = fast_metric(new_h)
                    if v < val_curr - 1e-12:
                        h_curr = new_h
                        val_curr = v
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                if val_curr < best_val:
                    best_val = val_curr
                    best_h = h_curr.copy()
                # Update peak lags periodically
                if _ % 50 == 0:
                    c = get_autoconv(h_curr)
                    peak_lags = np.argsort(c)[::-1][:top_k]
    
    # Phase 5: Fine-tune with small paired moves
    if elapsed() < DEADLINE - 5:
        step_frac = 0.005
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
    
    # Phase 6: Final high-precision polish with exact max and tiny steps
    if elapsed() < DEADLINE - 2:
        h_final = best_h.copy()
        val_final = best_val
        mean_h = np.mean(h_final)
        
        for _ in range(500):
            if elapsed() > DEADLINE - 1:
                break
            # Try a few very small single-coordinate adjustments
            for _ in range(3):
                idx = np.random.randint(N)
                for factor in [0.99, 0.995, 1.005, 1.01]:
                    new_h = h_final.copy()
                    new_h[idx] *= factor
                    if new_h[idx] < 0:
                        continue
                    v = fast_metric(new_h)
                    if v < val_final - 1e-14:
                        h_final = new_h
                        val_final = v
                        break
        
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
